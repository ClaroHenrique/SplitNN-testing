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

for model in models:
    for quantization in quantization_type:
        for split_point in split_points:
            for dataset, image_size, train_size in datasets__image_size__train_size:
                for batch_size in batch_sizes:
                    experiment_configs.append({
                        "MODEL": model,
                        "QUANTIZATION_TYPE": quantization,
                        "SPLIT_POINT": split_point,
                        "DATASET": dataset,
                        "IMAGE_SIZE": image_size,
                        "DATASET_TRAIN_SIZE": train_size,
                        "CLIENT_BATCH_SIZE": batch_size,
                    })


print("Experiment configurations:")
for config in experiment_configs:
    print(config)









