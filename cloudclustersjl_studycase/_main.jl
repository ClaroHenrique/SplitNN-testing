using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates
using CUDA

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")
include("src/models/get_model.jl")

learning_rate = 0.001
batch_size = 32
iterations_per_client = 200

# Define model
# model_name = "custom"
# model_name = "vgg16"
model_name = "resnet18"
# model_name = "mobilenetv3_small"
# model_name = "mobilenetv3_large"

model, img_dims = get_model(model_name)

# Download dataset
train_data = CIFAR10(split=:train)[:]
test_data = CIFAR10(split=:test)[:]

# Define data loaders
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

### Inicitalizate client nodes ###
# all nodes but master (id=1) are clients
println("Initializating clients")
addprocs(2)

@everywhere workers() begin
  using Distributed
  using Flux
  using ProgressLogging
  using CUDA

  # CUDA.device!(myid() % 2)   ***
  println("Initializating client $(myid())")
  # Load dependencies
  include("src/train_client.jl")
  include("src/dataset_loader.jl")

  # Create local data loader
  train_data = CIFAR10(split=:train)[:]
  train_loader = dataset_loader(train_data,
    batch_size=$batch_size,
    img_dims=$img_dims,
    n_batches=$iterations_per_client,
  )

  train_client(model) = train_client(model, train_loader)
end

println("Log initial test accuracy")
initial_timestamp = now()
log_model_accuracy(model |> gpu, partial_test_loader; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_iterations = 100

@profile @showprogress for ep in 1:num_epochs
  global model

  # Send global model to clients and start the training
  f_references = map(workers()) do client_id
    remotecall(train_client, client_id, model)
  end
  
  # Collect the result of the client training
  local_models = map(f_references) do client_model_future_ref
    fetch(client_model_future_ref)
  end

  # Aggregate clients' results
  model = aggregate(local_models) 

  # Print partial accuracy
  log_model_accuracy(model |> gpu, partial_test_loader; epoch=ep, timestamp=now() - initial_timestamp)
end

println("Full test accuracy:")
log_model_accuracy(model |> gpu, test_loader; epoch=ep, timestamp=now() - initial_timestamp)
