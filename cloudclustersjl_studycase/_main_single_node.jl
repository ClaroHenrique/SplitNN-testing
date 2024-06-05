import Pkg
Pkg.instantiate()

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
num_interations = 100

# Define model
# model_name = "custom"
# model_name = "vgg16"
model_name = "resnet18"
# model_name = "mobilenetv3_small"
# model_name = "mobilenetv3_large"

include("src/models/get_model.jl")
model, img_dims = get_model(model_name)
model = model |> gpu

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

optimizer = Flux.setup(Flux.Optimisers.Adam(learning_rate), model) |> gpu

# Log model initial test accuracy
initial_timestamp = now()
log_model_accuracy(model, test_loader; iteration=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")

@profile @showprogress for it in 1:num_interations
  Flux.train!(model, train_loader, optimizer) do m, x, y
    y_hat = m(x)
    Flux.logitcrossentropy(y_hat, y)
  end

  log_model_accuracy(model, partial_test_loader; iteration=it, timestamp=now() - initial_timestamp)
end

println("Complete test accuracy")
log_model_accuracy(model, partial_test_loader; iteration=num_interations, timestamp=now()-initial_timestamp)
