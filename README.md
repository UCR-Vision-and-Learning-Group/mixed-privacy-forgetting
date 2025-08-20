# mixed-privacy-forgetting

> implementation used for: **Towards Source-Free Machine Unlearning** (CVPR 2025), [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Ahmed_Towards_Source-Free_Machine_Unlearning_CVPR_2025_paper.html)

## citation
```bibtex
@inproceedings{ahmed_2025_cvpr,
    author    = {Ahmed, Sk Miraj and Basaran, Umit Yigit and Raychaudhuri, Dripta S. and Dutta, Arindam and Kundu, Rohit and Niloy, Fahim Faisal and Guler, Basak and Roy-Chowdhury, Amit K.},
    title     = {Towards Source-Free Machine Unlearning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4948-4957}
}
```

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
