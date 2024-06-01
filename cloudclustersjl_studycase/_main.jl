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
#model = fmap(cu, model)

# Define data loaders
test_data = CIFAR10(split=:test)[:]
test_loader = dataset_loader(test_data,
  img_dims=img_dims,
  n_batches=10, #TODO: How many test batches?
)

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

  # Load dependencies
  include("src/train_client.jl")
  include("src/dataset_loader.jl")

  # Define global constants
  learning_rate = 0.001
  batch_size = 32
  iterations_per_client = 100

  # Create local data loader
  train_data = CIFAR10(split=:train)[:]
  train_loader = dataset_loader(train_data,
    batch_size=batch_size,
    img_dims=$img_dims,
    n_batches=iterations_per_client,
  )

  train_client(model) = train_client(model, train_loader)
end

# Log model initial test accuracy
initial_timestamp = now()
log_model_accuracy(model, test_loader; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_epochs = 100

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

  #local_models = fmap(cu,local_models)

  # Aggregate clients' results

  #  global_model = aggregate(local_models) 
#  log_test_accuracy(global_model; epoch=ep, timestamp=now() - initial_timestamp) 

  model = aggregate(local_models) 
  log_model_accuracy(model, test_loader; epoch=ep, timestamp=now() - initial_timestamp)
end
