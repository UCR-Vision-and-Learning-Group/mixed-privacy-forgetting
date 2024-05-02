# Mixed Privacy Forgetting

## pretrain model

```bash
python main.py --mode pretrain --arch-id resnet50 --dataset-id cifar10 --split-rate 0.5
```

## train mixed model 

```bash
python main.py --mode train-user-data --arch-id resnet50 --dataset-id cifar10 --number-of-linearized-components 5 --use-default
```
