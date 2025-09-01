python train_sfl_local.py \
    --model_name MobileNetV2 \
    --quantization_type ptq \
    --dataset_name Cifar10 \
    --split_point 1 \
    --num_clients 1 \
    --image_size 224 224 \
    --client_batch_size 128 \
    --learning_rate 0.001 \
    --epochs 10