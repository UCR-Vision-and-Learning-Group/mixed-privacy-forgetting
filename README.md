# Mixed Privacy Forgetting

## pretrain model

```bash
python main.py --mode pretrain --arch-id resnet50 --dataset-id cifar10 --split-rate 0.5
```

## train mixed model

```bash
python main.py --mode train-user-data --arch-id resnet50 --dataset-id cifar10 --number-of-linearized-components 5 --use-default
```

```bash
python main.py --mode train-user-data --arch-id resnet50 --dataset-id cifar10 --number-of-linearized-components 5 --pretrained-model-path checkpoint/05022024-112451-pretrain-resnet50-cifar10-split0.5/05022024_112451_pretrain_resnet50_cifar10_split0.5.pth --split-rate 0.5
```

```bash
python main.py --mode train-user-data --arch-id resnet50 --dataset-id cifar10 --number-of-linearized-components 5 --split-rate 0.1 --use-default --device-id 1
```
