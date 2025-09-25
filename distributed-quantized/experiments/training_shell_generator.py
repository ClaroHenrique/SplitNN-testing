import itertools
import csv

dataset_names = ["Cifar10_IID", "Cifar10_non_IID"] #["Cifar10_IID", "Cifar10_non_IID", "XRay_IID", "XRay_non_IID"]
model_names = ["ResNet18", "MobileNetV2"]
quantization_types = ["ptq", "qat"]
optimizers = ["Adam"]
split_points = [1,2,3] 
num_clients = [4,8,16]
image_sizes = [(224,224)]
server_batch_sizes = [128]
learning_rates = [0.0001]
epochs = [200]

log_file = "./experiments/training_log.txt"
result = f"echo 'Init logging...' > f{log_file} \n"
count = 0
skipped = 0

ignore_finished = True

# Columns that define an unique experiment
important_columns = [
    "model_name", "quantization_type", "split_point",
    "num_clients", "dataset_name", "optimizer_name", "learning_rate",
]

configs_done = []

with open("experiments/training_results.csv", "r", newline="") as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        if int(linha["epoch"]) == 200:
            config = dict([(k, v) for k,v in linha.items() if k in important_columns])
            configs_done.append(config)


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
    config = {}
    config["model_name"] = model
    config["quantization_type"] = quant
    config["split_point"] = str(split)
    config["num_clients"] = str(n_clients)
    config["dataset_name"] = dataset
    config["optimizer_name"] = opt
    config["learning_rate"] = str(lr)

    if ignore_finished and config in configs_done:
        print("Skipping already finished experiment: ", config)
        skipped += 1
        continue

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

#print("Generated ", count, "experiments commands of a total of ", count+skipped)
print(f"Generated {count} experiments commands of a total of {count+skipped} ({count/(count+skipped)*100:.1f}% new).")
