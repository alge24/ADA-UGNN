# A Unified View on Graph Neural Networks as Graph Signal Denoising
Pytorch implementation of [ADA-UGNN](https://arxiv.org/pdf/2010.01777.pdf). Some parts of the code are adapdted from this [repo](https://github.com/zhao-tong/GNNs-easy-to-use).

For more details of the algorithm, please refer to our [paper](https://arxiv.org/pdf/2010.01777.pdf). If you find this work useful and use it in your research, please cite our paper.

```
@article{ma2020unified,
  title={A unified view on graph neural networks as graph signal denoising},
  author={Ma, Yao and Liu, Xiaorui and Zhao, Tong and Liu, Yozen and Tang, Jiliang and Shah, Neil},
  journal={arXiv preprint arXiv:2010.01777},
  year={2020}
}

```

### Requirements
```
torch               1.4.0+cu100
torchvision         0.5.0+cu100
networkx            2.5
numpy               1.19.1
```

### Usage
All the hyper-parameters settings are included in the [`run.sh`](https://github.com/alge24/ADA-UGNN/blob/main/code/run.sh) file. 
