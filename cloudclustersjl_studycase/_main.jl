using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates
#using CUDA
 
include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")

### Inicitalizate client nodes ###

# all nodes but master (id=0) are clients
println("Initializating clients")
addprocs(4)

@everywhere workers() begin
  #using Pkg; Pkg.activate(@__DIR__)
  #Pkg.instantiate(); Pkg.precompile()

  using Distributed
  using Flux
  using CUDA  

  # CUDA.device!(myid() % 2)   ***

  # Load dependencies
  include("src/train_client.jl")
  include("src/dataset_loader.jl")

  # Define global constants
  learning_rate = 0.001
  batch_size = 32
  iterations_per_client = 400

  # Create local data loader
  data_loader = dataset_loader(batch_size) 
end

# Define model
global_model = custom_model

#include("src/models/vgg16.jl")
#global_model = vgg16


# Log model initial test accuracy
initial_timestamp = now()
log_test_accuracy(global_model; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_epochs = 100

@profile @showprogress for ep in 1:num_epochs
  global global_model

  # Send global model to clients and start the training
  f_references = map(workers()) do client_id
    remotecall(train_client, client_id, global_model)
  end

  # Collect the result of the client training
  local_models = map(f_references) do client_model_future_ref
    fetch(client_model_future_ref)
  end

  #local_models = fmap(cu,local_models)

  # Aggregate clients' results
  global_model = aggregate(local_models) 
  log_test_accuracy(global_model; epoch=ep, timestamp=now()-initial_timestamp) 
end
