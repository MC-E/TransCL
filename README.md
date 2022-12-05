# TransCL: Transformer Makes Strong and Flexible Compressive Learning (TPAMI 2022)
> [[Paper](https://ieeexplore.ieee.org/document/9841016)]<br>

## The source code and models will be released soon. 

<p align="center">
  <img src="figs/network.PNG" width="80%">
</p>

## Training

```bash
Training on ImageNet

python main_imagenet.py -a 'B_32_imagenet1k' -b 128 --image_size 384 --gpu 0  --lr 1e-3 --log_dir logs/transcl_384_imagenet_p32_01 --cs=1 --mm=1 --save_path=transcl_384_imagenet_p32_01 --devices=4 --rat 0.1
```

## Testing

```bash
Testing on ImageNet

python test.py -a 'B_32_imagenet1k' -b 128 --image_size 384
```

## :european_castle: Model Zoo

| Pre-trained ViT        |                           URL                     |  
| :------------------- | :--------------------------------------------: |
| ImageNet Classification        |                           URL                     |  
| Cifar10 Classification        |                           URL                     |  
| Cifar100 Classification        |                           URL                     |  
| Arbitrary Ratio Classification        |                           URL                     | 
| Binary Sampling Classification        |                           URL                     | 
| Shuffled Classification        |                           URL                     | 

