import itertools

dataset_names = ["Cifar10_IID", "Cifar10_non_IID"] #["Cifar10_IID", "Cifar10_non_IID", "XRay_IID", "XRay_non_IID"]
model_names = ["ResNet18", "MobileNetV2"]
quantization_types = ["ptq", "qat"]
optimizers = ["Adam"]
split_points = [1,2,3] # [1,2,3]
num_clients = [4,8]
image_sizes = [(224,224)]
server_batch_sizes = [128]
learning_rates = [0.0001]
epochs = [200]

log_file = "./experiments/training_log.txt"
result = f"echo 'Init logging...' > f{log_file} \n"
count = 0

for (dataset, model, quant, opt, split, n_clients, img_size, server_batch_size, lr, ep) in itertools.product(
        dataset_names,
        model_names,
        quantization_types,
        optimizers,
        split_points,
        num_clients,
        image_sizes,
        server_batch_sizes,
        learning_rates,
        epochs,
        ):
    client_batch_size = server_batch_size // n_clients
    cmd = (
        f"python3 train_sfl_local.py "
        f"--model_name {model} "
        f"--quantization_type {quant} "
        f"--optimizer {opt} "
        f"--dataset_name {dataset} "
        f"--split_point {split} "
        f"--num_clients {n_clients} "
        f"--image_size {img_size[0]} {img_size[1]} "
        f"--client_batch_size {client_batch_size} "
        f"--learning_rate {lr} "
        f"--epochs {ep} "
        f">> {log_file}\n\n"
    )
    result += cmd
    count += 1

print(result)
with open('./experiments/training_script.sh', 'w') as f:
    f.write(result)

print("Total of: ", count, "experiments")
