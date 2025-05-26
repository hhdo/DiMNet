import sys
import torch
from tqdm import tqdm
import random
sys.path.append("..")
import utils
from torch import nn
import json
from config import args
from model import DiMNet
from datetime import datetime
from torch.utils import data as torch_data
from torch import distributed as dist
import wandb


def train_and_validate(args, model, train_list, valid_list, test_list, num_nodes, num_rels, model_state_file):
    print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart training\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    best_mrr = 0
    patience = args.patience
    for epoch in range(args.n_epoch):
        if patience == 0:
            print("Early stopping at epoch: {}".format(epoch))
            break
        # print("\nepoch:"+str(epoch)+ ' Time: ' + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'))
        model.train()
        losses = list()
        cllosses = list()
        predlosses = list()

        idx = [_ for _ in range(len(train_list))] # timestamps index [0,1,2,3,...,n]
        random.shuffle(idx)
        idx_proc = tqdm(idx, ncols=100, desc='Epoch %i'%epoch)
        for future_sample_id in idx_proc:
        # for future_sample_id in idx:
            if future_sample_id == 0: continue
            # future_sample as the future graph
            futrue_graph = train_list[future_sample_id]
            # future_triple : [num_edges, 3] (format: h,t,r)
            # Note that we also add reverse edges in 'future_triple' as query query_triple
            future_triple = torch.cat((futrue_graph.edge_index, futrue_graph.edge_type.unsqueeze(0))).t()
            
            # get history graph list
            if future_sample_id - args.history_len < 0:
                history_list = train_list[0: future_sample_id]
            else:
                history_list = train_list[future_sample_id - args.history_len:
                                    future_sample_id]
            

            batch = future_triple # all future tirples is an only batch   
            
            
            pred, cl_loss = model(history_list, batch)
            pred_loss = model.get_loss(pred, batch[:,1])
            loss = pred_loss + cl_loss

            predlosses.append(pred_loss.item())
            cllosses.append(cl_loss.item() if cl_loss != 0 else 0.0)        
            losses.append(loss.item())
            
            # idx_proc.set_postfix(loss='%f'%(sum(losses) / len(losses)))
            idx_proc.set_postfix(pred_loss = '%f'%(sum(predlosses) / len(predlosses)), 
                                 cl_loss = '%f'%(sum(cllosses) / len(cllosses)))
            wandb.log({"pred_loss": sum(predlosses) / len(predlosses), 
                       "cl_loss": sum(cllosses) / len(cllosses),
                       "loss": sum(losses) / len(losses)})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
                        

        # evaluation
        print("valid dataset eval:", end='\t')
        metrics_valid = test(model, valid_list, num_rels, num_nodes)
        wandb.log({"MRR": metrics_valid['mrr'], 
                   "H1": metrics_valid['hits@1'], 
                   "H3": metrics_valid['hits@3'], 
                   "H10": metrics_valid['hits@10'],
                   "epoch": epoch})

        if metrics_valid['mrr'] >= best_mrr:
            best_mrr = metrics_valid['mrr']
            patience = args.patience
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args}, model_state_file)
            print("--------best_mrr updated!--------")
            
        else:
            patience -= 1    
            print("---------------------------------")
        
    
    # testing
    
    print("\nFinal eval test dataset with best model:...")
    metrics_test = test(model, test_list, num_rels, num_nodes, mode="test", model_name=model_state_file)
    wandb.log({"MRR-test": metrics_test['mrr'], 
               "H1-test": metrics_test['hits@1'], 
               "H3-test": metrics_test['hits@3'], 
               "H10-test": metrics_test['hits@10']})

    return best_mrr

@torch.no_grad()
def test(model, test_list, num_rels, num_nodes, mode="train", model_name = None):

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'], checkpoint['args']))  # use best stat checkpoint
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    idx = [_ for _ in range(len(test_list))] # timestamps index [0,1,2,3,...,n]

    model.eval()
    rankings = []

    for future_sample_id in idx:
        if future_sample_id < args.history_len: continue
        # future_sample as the future graph
        future_graph = test_list[future_sample_id]
        # future_triple : [num_edges, 3] (format: h,t,r)
        # Note that we are not add reverse edges in 'future_triple' as query query_triple in test phase
        future_triple = future_graph.target_triplets
        future_triple_reverse = future_triple[:, [1,0,2]]
        future_triple_reverse[:,2] += num_rels
        
        # get history graph list
        history_list = test_list[future_sample_id - args.history_len : future_sample_id]

        # time_filter data only contains the future triple without reverse edges for mask generation
        time_filter_data = {
                'num_nodes': num_nodes,
                'edge_index': torch.stack([future_triple[:,0], future_triple[:,1]]),
                'edge_type': future_triple[:,2]
        }
        
        batch = future_triple # all future tirples is an only batch 
        
        triple = torch.cat([future_triple, future_triple_reverse])
        pred, _ = model(history_list, triple)
        
        t_pred, h_pred = torch.chunk(pred, 2, dim=0)
        pos_h_index, pos_t_index, pos_r_index = batch.t()


        # time_filter Rank
        timef_t_mask, timef_h_mask = utils.strict_negative_mask(time_filter_data, batch) 
        timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask)
        timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask)
        rankings += [timef_t_ranking, timef_h_ranking]

        # This is the end of prediction at 'future_sample_id' time
    # This is the end of prediction at test_set

    all_ranking = torch.cat(rankings)
    
    metrics_dict = dict()
    for metric in args.metric:
        if metric == "mr":
            score = all_ranking.float().mean()
        elif metric == "mrr":
            score = (1 / all_ranking.float()).mean()
        elif metric.startswith("hits@"):
            values = metric[5:].split("_")
            threshold = int(values[0])
            score = (all_ranking <= threshold).float().mean()
        metrics_dict[metric] = score.item()*100
    metrics_dict['time'] = datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')
    # print(json.dumps(metrics_dict, indent=4))
    print("MRR:{:.5f}\tH1:{:.5f}\tH3:{:.5f}\tH10:{:.5f}"
          .format(metrics_dict['mrr'],metrics_dict['hits@1'],metrics_dict['hits@3'],metrics_dict['hits@10']))
        
    # mrr = (1 / all_ranking.float()).mean()

    return metrics_dict


if __name__ == '__main__':
    current_timestamp = datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')
    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args, current_timestamp)
    device = utils.get_device(args)
    
    model_name = "len:{}-dim:{}-ly:{}-head:{}-topk:{}-"\
        .format(args.history_len, args.input_dim, args.num_ly, args.num_head, 
                args.topk)

    model_state_file = model_name+current_timestamp+".pth"

    # load datasets
    data = utils.load_data(args.dataset)
    num_nodes = data.num_nodes
    num_rels = data.num_rels # not include reverse edge type

    
    print("# Model ID: {}".format(model_state_file))
    print("# Sanity Check: entities: {}".format(data.num_nodes))
    print("# Sanity Check: relations: {}".format(data.num_rels))
    print("# Sanity Check: edges: {}".format(len(data.train)))


    train_graph_list, valid_graph_list, test_graph_list = \
        utils.generate_graph_data(data, num_nodes, num_rels, False, device)
    # Each item in the graph list is a snapshot of the graph
    # edge_index: [2, num_edges], which has added reverse edges
    # edge_type: [num_edges], which has added reverse edges
    # num_nodes: int
    # target_triplets: [num_edges, 3] (format: h,t,r)

    train_list = train_graph_list
    valid_list = train_list[-args.history_len:] + valid_graph_list
    test_list = valid_list[-args.history_len:] + test_graph_list

    model = DiMNet(
        dim=args.input_dim,
        num_layer=args.num_ly,
        num_relation=num_rels,
        num_node=num_nodes,
        message_func=args.message_func, 
        aggregate_func=args.aggregate_func,
        short_cut=args.short_cut, 
        layer_norm=args.layer_norm,
        activation="rrelu", 
        history_len=args.history_len,
        topk=args.topk,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        num_head=args.num_head
    )
    model = model.to(device)
    wandb.watch(model, log='all', log_freq=500)
    if args.test:
        test(model, test_list, num_rels, num_nodes, mode="test", model_name = model_state_file)
    else:
        train_and_validate(args, model, train_list, valid_list, test_list, num_nodes, num_rels, model_state_file)

    sys.exit()



