# [NeurIPS 2024] Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization

Code release for **Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization (NeurIPS 2024)**.

[[paper]](https://arxiv.org/abs/2405.16173) [[project page]](#)

![](./imgs/QVPO_pipeline.jpg)

## Requirements
```

```
## Preparation

To run the experiments, you need to first install the python package rpo via running `pip install -e .` in the current directory.

## Getting started
Then, you can simply 

run `python scripts/cart_exp.py` to re-implement our experiments on the Safe Cartpole environment using RPODDPG algorithm.

run `python scripts/pen_exp.py` to re-implement our experiments on the Spring Pendulum environment using RPODDPG algorithm.

run `python scripts/evopf_exp.py` to re-implement our experiments on the OPF with Battery Energy Storage environment using RPODDPG algorithm.

run `python scripts/cart_exp_sac.py` to re-implement our experiments on the Safe Cartpole environment using RPOSAC algorithm.

run `python scripts/pen_exp_sac.py` to re-implement our experiments on the Spring Pendulum environment using RPOSAC algorithm.

run `python scripts/evopf_exp_sac.py` to re-implement our experiments on the OPF with Battery Energy Storage environment using RPOSAC algorithm.

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{
anonymous2024diffusionbased,
title={Diffusion-based Reinforcement Learning via Q-weighted Variational Policy Optimization},
author={Shutong Ding and Ke Hu and Zhenhao Zhang and Kan Ren and Weinan Zhang and Jingyi Yu, Jingya Wang and Ye Shi},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=UWUUVKtKeu}
}
