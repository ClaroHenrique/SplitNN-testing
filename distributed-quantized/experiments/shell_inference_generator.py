import itertools

dataset_names = ["Cifar10_IID"]
model_names = ["ResNet18", "MobileNetV2"]
quantization_types = ["ptq"]
split_points = [1,2,3] # [1,2,3]
num_clients = [1]
image_sizes = [(224,224)]
client_batch_sizes = [128]

file = "./experiments/inference_configs.csv"

all_configs = itertools.product(
    dataset_names,
    model_names,
    quantization_types,
    split_points,
    num_clients,
    image_sizes,
    client_batch_sizes,
)
columns = [
    "dataset_names",
    "model_names",
    "quantization_types",
    "split_points",
    "num_clients",
    "image_sizes",
    "client_batch_sizes",
]
result = ",".join(columns) + '\n'
count = 0

for (dataset, model, quant, split, n_clients, img_size, client_batch_size) in all_configs:
    exp_config = (
        f"{model}, "
        f"{quant}, "
        f"{dataset}, "
        f"{split}, "
        f"{n_clients}, "
        f"{img_size[0]} {img_size[1]}, "
        f"{client_batch_size}, "
        "\n"
    )
    result += exp_config
    count += 1


print(result)
with open(file, 'w') as f:
    f.write(result)
print("Total of: ", count, "experiments")
