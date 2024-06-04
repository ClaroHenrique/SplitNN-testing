using ProgressMeter
using Profile
using CUDA, cuDNN
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")

### Inicitalizate client nodes ###
println("Initializating")

# Define constants
learning_rate = 0.001
batch_size = 32
iterations_per_client = 100

# Define model
# model_name = "custom"
# model_name = "vgg16"
model_name = "resnet18"
# model_name = "mobilenetv3_small"
# model_name = "mobilenetv3_large"

include("src/models/get_model.jl")
model, img_dims = get_model(model_name)
model |> gpu

# Download dataset
train_data = CIFAR10(split=:train)[:]
test_data = CIFAR10(split=:test)[:]

# Define data loaders
train_loader = dataset_loader(train_data,
  batch_size=batch_size,
  img_dims=img_dims,
  n_batches=iterations_per_client,
)
test_loader = dataset_loader(test_data,
  batch_size=batch_size,
  img_dims=img_dims,
  n_batches=10000,
)
partial_test_loader = dataset_loader(test_data,
  batch_size=batch_size,
  img_dims=img_dims,
  n_batches=iterations_per_client,
) # Run test using less instances


# Log model initial test accuracy
initial_timestamp = now()
log_model_accuracy(model, test_loader; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_epochs = 100

@profile @showprogress for ep in 1:num_epochs
  global model
  train_client(model, train_loader)
  log_model_accuracy(model, partial_test_loader; epoch=ep, timestamp=now()-initial_timestamp)
end

println("Complete test accuracy")
log_model_accuracy(model, partial_test_loader; epoch=ep, timestamp=now()-initial_timestamp)