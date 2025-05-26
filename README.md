# Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning

This is the official code release of the following paper:

Hao Dong, Ziyue Qiao, Zhiyuan Ning, Qi Hao, Yi Du, Pengyang Wang and Yuanchun Zhou. "[Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2505.14020)." ACL 2025 Findings.

<img src="https://github.com/hhdo/DiMNet/blob/main/img/DiMNet.png" alt="DiMNet_Architecture" width="800" class="center">

## Quick Start

### Dependencies

```
conda create -n dimnet python=3.8.15
conda activate dimnet
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3  -c pytorch  -c conda-forge
conda install pytorch-geometric==2.0.4 -c rusty1s -c conda-forge
```

❗️**Note**: This work also requires the *wandb* environment to be set up in advance and logged in. (Ref: [W&B Quickstart](https://docs.wandb.ai/quickstart/))

### Train & Evaluate Models

1. Switch to `src/` folder
```
cd src/
``` 

2. Run scripts

```
python main.py -d ICEWS18 --history_len 10 --num_head 4 --num_ly 3 --topk 50 --decay 1e-4 --gpu 0
```


### Change the Hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other setting according to the ***Implementation Details*** section in the [paper](https://arxiv.org/pdf/2505.14020). 

## Citation
If you find the resource in this repository helpful, please cite

```bibtex
@article{dong2025disentangled,
  title={Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning},
  author={Dong, Hao and Qiao, Ziyue and Ning, Zhiyuan and Hao, Qi and Du, Yi and Wang, Pengyang and Zhou, Yuanchun},
  journal={arXiv preprint arXiv:2505.14020},
  year={2025}
}
```