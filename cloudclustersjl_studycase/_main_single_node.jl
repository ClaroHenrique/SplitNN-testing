using ProgressMeter
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates
#using CUDA

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
model_name = "custom"
#model_name = "vgg16"
if model_name == "custom"
  include("src/models/custom.jl")
  model = custom_model
  img_dims = (32,32)
elseif model_name == "vgg16"
  include("src/models/vgg16.jl")
  model = vgg16
  img_dims = (224, 224)
end
model = fmap(cu, model)

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
  n_batches=iterations_per_client,
)

# Log model initial test accuracy
initial_timestamp = now()
log_model_accuracy(model, test_loader; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_epochs = 100

@profile @showprogress for ep in 1:num_epochs
  global model
  train_client(model, train_loader)
  log_model_accuracy(model, test_loader; epoch=ep, timestamp=now()-initial_timestamp)
end
