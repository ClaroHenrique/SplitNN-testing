python train_sfl_local.py \
    --model_name ResNet18 \
    --quantization_type ptq \
    --dataset_name ImageNette \
    --split_point 1 \
    --num_clients 1 \
    --image_size 224 224 \
    --client_batch_size 128 \
    --learning_rate 0.001 \
    --epochs 10