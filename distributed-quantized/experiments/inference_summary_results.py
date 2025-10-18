import itertools
import csv

def average(lst):
    if not lst:
        raise ValueError("Empty list to compute average")
    return sum(lst) / len(lst) if lst else 0

def stddev(lst):
    if len(lst) < 2:
        raise ValueError("Stddev requires at least two elements")
    avg = average(lst)
    variance = sum((x - avg) ** 2 for x in lst) / (len(lst) - 1)
    return variance ** 0.5

# Columns that define an unique experiment
important_columns = [
    "model_name", "split_point", "dataset_name", "batch_size"
]

# run_id,model_name,quantization_type,split_point,dataset_name,batch_size,mem-peak-mb,mem-first-mb,mem-usage-mb,time,bandwidth,model-size

#Columns to remove from the summary file
metric_columns = [
    "mem-usage-mb", "time", "bandwidth", "model-size",
]
configs_done = {}

dict_mems = {}
dict_times = {}

dict_mems_quant = {}
dict_times_quant = {}

with open("experiments/inference_results.csv", "r", newline="") as f:
    leitor = csv.DictReader(f)
    for linha in leitor:
        key = tuple((linha[col] for col in important_columns))
        if key not in configs_done:
            dict_mems[key] = []
            dict_times[key] = []
            dict_mems_quant[key] = []
            dict_times_quant[key] = []
            configs_done[key] = dict((col, linha[col]) for col in important_columns)

        if linha["quantization_type"] == "full":
            configs_done[key]["model-size"] = linha["model-size"]
            configs_done[key]["bandwidth"] = linha["bandwidth"]
            dict_mems[key].append(float(linha["mem-usage-mb"]))
            dict_times[key].append(float(linha["time"]))
        else:
            configs_done[key]["model-size-quant"] = linha["model-size"]
            configs_done[key]["bandwidth-quant"] = linha["bandwidth"]
            dict_mems_quant[key].append(float(linha["mem-usage-mb"]))
            dict_times_quant[key].append(float(linha["time"]))


for key, config in configs_done.items():
    # full
    mems = dict_mems[key]
    times = dict_times[key]

    config["time-avg"] = average(times)
    config["time-std"] = stddev(times)
    config["mem-usage-mb-avg"] = average(mems)
    config["mem-usage-mb-std"] = stddev(mems)

    # quantized
    mems_quant = dict_mems_quant[key]
    times_quant = dict_times_quant[key]

    config["time-avg-quant"] = average(times_quant)
    config["time-std-quant"] = stddev(times_quant)
    config["mem-usage-mb-avg-quant"] = average(mems_quant)
    config["mem-usage-mb-std-quant"] = stddev(mems_quant)

    config["sample-size"] = len(mems)


print("configs_done:", configs_done)

# save configs_done to a csv file
with open("experiments/inference_summary_results.csv", "w", newline="") as f:
    keys = next(iter(configs_done.values())).keys()
    escritor = csv.DictWriter(f, fieldnames=keys)
    escritor.writeheader()
    for config in configs_done.values():
        escritor.writerow(config)

