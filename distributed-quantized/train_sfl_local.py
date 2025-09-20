import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import copy
from optimizer.adam import create_optimizer
from model.quantization import generate_quantized_model
from torch import nn
from utils.utils import *

#####################################################
## define Variables
import argparse
parser = argparse.ArgumentParser(description='Script para treinamento e quantização de modelos de visão.')

parser.add_argument('--model_name', type=str, required=True, help='Nome do modelo (ex: ResNet18, MobileNetV2)')
parser.add_argument('--quantization_type', type=str, required=True, help='Tipo de quantização (ex: ptq, qat)')
parser.add_argument('--optimizer_name', type=str, required=True, help='Nome do otimizador (ex: Adam, SGD)')
parser.add_argument('--dataset_name', type=str, required=True, help='Nome do dataset (ex: imagenette, cifar10)')
parser.add_argument('--split_point', type=int, required=True, help='Ponto de divisão para aprendizagem federada')
parser.add_argument('--num_clients', type=int, required=True, help='Número de clientes na aprendizagem federada')
parser.add_argument('--image_size', type=int, nargs=2, required=True, help='Tamanho da imagem como altura e largura (ex: 224 224)')
parser.add_argument('--client_batch_size', type=int, required=True, help='Tamanho do batch para o cliente')
parser.add_argument('--learning_rate', type=float, required=True, help='Taxa de aprendizado')
parser.add_argument('--epochs', type=int, required=True, help='Número de épocas de treinamento')
parser.add_argument('--has_to_load_model', action='store_true', help='Carrega estado do modelo salvo anteriormente.')
parser.add_argument('--inference_only', action='store_true', help='Realiza apenas a inferência, sem treino.')

args = parser.parse_args()

model_name = args.model_name
quantization_type = args.quantization_type
optimizer_name = args.optimizer_name
dataset_name = args.dataset_name
split_point = args.split_point
num_clients = args.num_clients
image_size = tuple(args.image_size)
client_batch_size = args.client_batch_size
learning_rate = args.learning_rate
epochs = args.epochs
has_to_load_model = args.inference_only
inference_only = args.inference_only
results_file_name = "results_accuracy.csv"
run_id = generate_run_id()
print("inference_only", inference_only)


device_client = torch.device("cuda" if torch.cuda.is_available() and quantization_type != 'qat' else "cpu")
device_server = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

#####################################################
# Define Data
from dataset.datasets import get_data_loaders
print("Starting data loaders")
train_calib_val_data_loaders = [get_data_loaders(dataset_name, batch_size=client_batch_size, client_id=client_id, num_clients=num_clients, image_size=image_size) for client_id in range(num_clients)]
train_data_loaders = [train_data_loader for train_data_loader, _, _ in train_calib_val_data_loaders]
calib_data_loaders = [calib_data_loader for _, calib_data_loader, _ in train_calib_val_data_loaders]
train_iters = [iter(train_data_loader) for train_data_loader in train_data_loaders]
val_loader = train_calib_val_data_loaders[0][2]  # Use the first client's test data loader for testing
def get_next_train_batch(client_id):
    global train_iters
    batch = next(train_iters[client_id], None)
    if batch is None:
        train_iters[client_id] = iter(train_data_loaders[client_id])
        return next(train_iters[client_id])
    return batch

#####################################################
## Define models
from model.models import ClientModel
from model.models import ServerModel
from dataset.datasets import get_num_classes
print("Starting models")

num_classes = get_num_classes(dataset_name)
init_server_model = ServerModel(model_name, num_classes, quantization_type, split_point=split_point, device=device_server, input_shape=None)
init_client_model = ClientModel(model_name, num_classes, quantization_type, split_point=split_point, device= device_client, input_shape=image_size)

client_models = [copy.deepcopy(init_client_model) for _ in range(num_clients)]
server_model = copy.deepcopy(init_server_model)
# Load models parameters
if has_to_load_model:
    load_model_if_exists(server_model, model_name, quantization_type, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
    for client_model in client_models:
        load_model_if_exists(client_model, model_name, quantization_type, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)
# Create optimizers
from optimizer.optimizers import get_optimizer_scheduler
server_optimizer, server_scheduler = get_optimizer_scheduler(optimizer_name, server_model.parameters(), learning_rate, epochs)
client_optimizers_schedulers = [get_optimizer_scheduler(optimizer_name, client_model.parameters(), learning_rate, epochs) for client_model in client_models]
client_optimizers = [optimizer for optimizer, _ in client_optimizers_schedulers]
client_schedulers = [scheduler for _, scheduler in client_optimizers_schedulers]


#####################################################
## Utils
def aggregate_client_model_params(client_models):
    global_model = copy.deepcopy(client_models[0])
    for name, param in global_model.named_parameters():
        new_w = torch.zeros_like(param)
        for c_model in client_models:
            new_w += c_model.state_dict()[name]
        new_w /= num_clients
        global_model.state_dict()[name].copy_(new_w)
    global_model_state = global_model.state_dict()
    for client_model in client_models:
        client_model.load_state_dict(global_model_state)

def test_accuracy_split(client_model, server_model):
    client_model.eval()
    server_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device_client)
            intermediate_representation = client_model(inputs)
            intermediate_representation = intermediate_representation.to(device_server)
            outputs = server_model(intermediate_representation).to('cpu')
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def print_test_accuracy(client_model, server_model, quantized=False):
    client_model.eval()
    server_model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            if quantized:
                inputs = inputs.to('cpu')
                tensor_IR = client_model(inputs)
                tensor_IR = tensor_IR.dequantize()
            else:
                inputs = inputs.to(device_client)
                tensor_IR = client_model(inputs)
            if device_client != device_server or quantized:
                tensor_IR = tensor_IR.to(device_server)
            outputs = server_model(tensor_IR).to('cpu')

            curr_loss = loss_fn(outputs, labels)
            pred_class = torch.argmax(outputs, dim=1)

            correct += torch.eq(labels, pred_class).sum().item()
            total += len(labels)
            loss += curr_loss.item()
        (quantized and (print("== Quantized Metrics ==") or True)) or print("== Full Precision Metrics ==")
        print(f"Accuracy: {correct} / {total} = {correct / total}")
        print(f"Loss: ", loss)
        print()
    return correct / total

def compare_full_and_quantized_model():
    client_model_quantized = generate_quantized_model(client_models[0], calib_data_loaders[0], quantization_type=quantization_type)
    full_acc = print_test_accuracy(client_model=client_models[0], server_model=server_model, quantized=False)
    quant_acc = print_test_accuracy(client_model=client_model_quantized, server_model=server_model, quantized=True)
    return full_acc, quant_acc

#####################################################
#### Inference

if inference_only:
    compare_full_and_quantized_model()
    print("Current (accuracy, loss)", test_accuracy_split(client_models[0], server_model))
    exit(0)

#####################################################
#### Train
print("Starting training")
start_time = time.time()
for ep in range(epochs+1):
    print()
    print(f"Epoch {ep}/{epochs}, LR: {server_optimizer.param_groups[0]['lr']:.8f}")
    if ep%10 == 0:
        full_acc, quant_acc = compare_full_and_quantized_model()
        print("Current (accuracy, loss)", test_accuracy_split(client_models[0], server_model))
        save_results_in_file(results_file_name, run_id, start_time, ep, full_acc, quant_acc, model_name, quantization_type, split_point, num_clients, dataset_name, optimizer_name, learning_rate)

    server_model.train()
    for client_model in client_models:
        client_model.train()
    for _ in range(200): # epoch: 200 iterations
        server_optimizer.zero_grad()
        for client_optimizer in client_optimizers:
            client_optimizer.zero_grad()
        outputs = []
        labels = []
        for client_id in range(num_clients):
            inputs, label = get_next_train_batch(client_id)
            IR = client_models[client_id](inputs.to(device_client))
            output = server_model(IR.to(device_server))
            outputs.append(output)
            labels.append(label)

        concat_outputs = torch.cat(outputs).to(device_server)
        concat_labels = torch.cat(labels).to(device_server)

        loss = loss_fn(concat_outputs, concat_labels)

        print("Loss", loss, end="\r")
        loss.backward()

        server_optimizer.step()
        for client_optimizer in client_optimizers:
            client_optimizer.step()

        aggregate_client_model_params(client_models)

    server_scheduler.step()
    for client_scheduler in client_schedulers:
        client_scheduler.step()
    save_state_dict(server_model.state_dict(), model_name, quantization_type, split_point, is_client=False, num_clients=num_clients, dataset_name=dataset_name)
    save_state_dict(client_models[0].state_dict(), model_name, quantization_type, split_point, is_client=True, num_clients=num_clients, dataset_name=dataset_name)

print("current accuracy, losss", test_accuracy_split(client_models[0], server_model))


#####################################################
## Evaluate and quantize
print("accuracy, losss", test_accuracy_split(client_models[0], server_model))
compare_full_and_quantized_model()
