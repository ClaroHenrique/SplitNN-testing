num_clients = [
    4,
    8,
]

models = [
    "ResNet18_custom",
    "ResNet34_custom",
]

quantization_type = [
    "ptq",
    "qat",
]

split_points = [
    1,
    2,
    3,
]

datasets__image_size__train_size = [
    ("cifar10_iid", (32,32), 50000),
    ("cifar10_non_iid", (32,32), 50000),
]

batch_sizes = [
    256,
]

experiment_configs = []

for num_client in num_clients:
    for model in models:
        for quantization in quantization_type:
            for split_point in split_points:
                for dataset, image_size, train_size in datasets__image_size__train_size:
                    for batch_size in batch_sizes:
                        experiment_configs.append({
                            "NUM_CLIENTS": num_client,
                            "MODEL_NAME": model,
                            "QUANTIZATION_TYPE": quantization,
                            "SPLIT_POINT": split_point,
                            "DATASET_NAME": dataset,
                            "IMAGE_SIZE": image_size,
                            "DATASET_TRAIN_SIZE": train_size,
                            "CLIENT_BATCH_SIZE": batch_size,
                        })

if __name__ == "__main__":
    print("Available configurations:")
    for config in experiment_configs:
        print(config)








