# train core dataset
python train.py --mode train-core-dataset --dataset-id cifar10 --arch-id resnet18 \
--split-rate 0.2 --batch-size 128 --criterion-id ce --optimizer-id adam \
--learning-rate 0.001 --num-epoch 20 --shuffle --use-pretrained \
--save-path ./checkpoints/042924-train-core-dataset/last_checkpoint.pth

