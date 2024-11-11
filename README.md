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
python main.py --mode train-user-data --arch-id resnet50 --dataset-id cifar10 --number-of-linearized-components 1 --pretrained-model-path checkpoint/05142024-180246-pretrain-resnet50-cifar10-split0.8/05142024_180246_pretrain_resnet50_cifar10_split0.8.pth --split-rate 0.8
```

```bash
python main.py --mode mixed-privacy --arch-id resnet18 --dataset-id cifar10-act -nlc 1 --split-rate 0.1 \ 
 -cp /home/umityigitbsrn/Desktop/umityigitbsrn/mixed-privacy-forgetting/checkpoint/05152024-011132-train-user-data-resnet18-cifar10-last1 \
 --activation-variant --device-id 0
```

## forget using remaining data (adahessian)

```bash
python main.py --mode forget-by-diag --arch-id resnet18 --dataset-id cifar10-act -nlc 1 --split-rate 0.1 \ 
 -cp /home/umityigitbsrn/Desktop/umityigitbsrn/mixed-privacy-forgetting/checkpoint/05152024-011132-train-user-data-resnet18-cifar10-last1 \
 --activation-variant --device-id 0 --num-iter-for-diag 500
```
