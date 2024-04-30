# train core dataset -- cifar10
python train.py --mode train-core-dataset --dataset-id cifar10 --arch-id resnet18 \
--split-rate 0.2 --batch-size 128 --criterion-id ce --optimizer-id adam \
--learning-rate 0.001 --num-epoch 20 --shuffle --use-pretrained \
--save-path ./checkpoints/042924-train-core-dataset-cifar10/last_checkpoint.pth

# train core dataset -- mnist
python train.py --mode train-core-dataset --dataset-id mnist --arch-id simple \
--split-rate 0.2 --batch-size 128 --criterion-id ce --optimizer-id adam \
--learning-rate 0.001 --num-epoch 20 --shuffle \
--save-path ./checkpoints/042924-train-core-dataset-mnist/last_checkpoint.pth \
--reshape-data -1 784


# train with jvp train dataset -- mnist
python train.py --mode train-with-jvp-train-dataset --dataset-id mnist --arch-id simple \
--checkpoint-path ./checkpoints/042924-train-core-dataset-mnist/last_checkpoint.pth \
--split-rate 0.2 --batch-size 128 --criterion-id mse --optimizer-id adam \
--learning-rate 0.001 --num-epoch 100 --shuffle \
--save-path ./checkpoints/042924-train-with-jvp-train-dataset-mnist/last_checkpoint.pth \
--reshape-data -1 784 --device-id 1 --weight-decay 0.001

# train with jvp train dataset -- cifar10
python train.py --mode train-with-jvp-train-dataset --dataset-id cifar10 --arch-id resnet18 \
--checkpoint-path ./checkpoints/042924-train-core-dataset-cifar10/last_checkpoint.pth \
--split-rate 0.2 --batch-size 128 --criterion-id mse --optimizer-id adam \
--learning-rate 0.001 --num-epoch 100 --shuffle \
--save-path ./checkpoints/042924-train-with-jvp-train-dataset-cifar10/last_checkpoint.pth \
--device-id 0 --weight-decay 0.001