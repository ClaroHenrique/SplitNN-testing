import itertools
import csv

log_file = "./experiments/training_log.txt"
result = f"echo 'Init logging...' >> {log_file} \n"
count = 0
skipped = 0

ignore_finished = True

# Columns that define an unique experiment
important_columns = [
    "model_name", "split_point",
    "num_clients", "dataset_name", "optimizer_name", 
]

#run_id,duration (s),epoch,full_accuracy,quant_accuracy,model_name,quantization_type,split_point,num_clients,dataset_name,optimizer_name,learning_rate

#Columns to remove from the summary file
columns_to_stay = [
    "model_name", "split_point",
    "num_clients", "dataset_name",
    "full_accuracy", "ptq_accuracy", "qat_accuracy",
]
configs_done = {}

# Read existing results, aggregating accuracies
with open("experiments/training_results.csv", "r", newline="") as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        epochs_done = int(linha["epoch"])
        quantization_type = linha["quantization_type"]
        if (quantization_type == "ptq" and epochs_done == 200) or (quantization_type == "qat" and epochs_done == 20):
            key = tuple((linha[col] for col in important_columns))
            if key not in configs_done:
                configs_done[key] = dict((col, linha[col]) for col in important_columns)
            if quantization_type == "ptq":
                configs_done[key]["full_accuracy"] = linha["full_accuracy"]
                configs_done[key]["ptq_accuracy"] = linha["quant_accuracy"]
            else: # qat
                configs_done[key]["qat_accuracy"] = linha["quant_accuracy"]
            
            configs_done[key]

# Remove unwanted columns and generate final list
final_configs = []
for config in configs_done.values():
    config = dict([col, value] for col, value in config.items() if col in columns_to_stay)
    final_configs.append(config)


# Add some more useful columns
columns_to_stay.append("relative_difference_ptq")
columns_to_stay.append("relative_difference_qat")

for config in final_configs:
    config["relative_difference_ptq"] = ""
    config["relative_difference_qat"] = ""
    if "ptq_accuracy" in config:
        config["relative_difference_ptq"] = float(config["ptq_accuracy"]) - float(config["full_accuracy"])
        config["relative_difference_ptq"] = f"{config['relative_difference_ptq']:.4f}"
    if "qat_accuracy" in config:
        config["relative_difference_qat"] = float(config["qat_accuracy"]) - float(config["full_accuracy"])
        config["relative_difference_qat"] = f"{config['relative_difference_qat']:.4f}"

# save configs_done to a csv file
with open("experiments/training_summary_results.csv", "w", newline="") as f:
    escritor = csv.DictWriter(f, fieldnames=columns_to_stay)
    escritor.writeheader()
    for config in final_configs:
        escritor.writerow(config)

