echo 'Init logging...' > f./experiments/training_log.txt 
python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name ResNet18 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type ptq --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 1 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 2 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 4 --image_size 224 224 --client_batch_size 32 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 8 --image_size 224 224 --client_batch_size 16 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

python3 train_sfl_local.py --model_name MobileNetV2 --quantization_type qat --optimizer Adam --dataset_name Cifar10_non_IID --split_point 3 --num_clients 16 --image_size 224 224 --client_batch_size 8 --learning_rate 0.0001 --epochs 200 >> ./experiments/training_log.txt

