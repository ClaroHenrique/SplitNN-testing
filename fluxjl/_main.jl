using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates

include("src/utils.jl")
include("src/aggregate.jl")

# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end

parameter_server_ids = addprocs(2)

@everywhere parameter_server_ids begin
  using Distributed
  using Flux
  include("src/train_parameter_server.jl")
  include("src/aggregate.jl")
  client_ids = addprocs(2)
  train_client(a,b) = nothing
  server_id = myid()
  
  ## Node initialization
  @everywhere client_ids begin
    using Distributed
    using Flux
    # Load dependencies
    include("src/dataset_loader.jl")
    include("src/train_client.jl")

    # Define global constants
    n_servers = 2
    n_clients_per_server = 2
    n_clients = n_servers * n_clients_per_server
    learning_rate = 0.001
    batch_size = 32
    batchs_per_epoch = 100

    # Load federated dataset
    server_id = @fetchfrom role=:worker 1 myid(role=:worker)
    data_id = (server_id-2)*2 + myid(role=:worker) - 1
    data_loader = dataset_loader(id=data_id, n_clients=n_clients, batch_size=batch_size)
  end
end

#include("src/dataset_loader.jl")
num_epochs = 20

# Define model
include("src/models/custom.jl")
global_model = custom_model

function log_model_accuracy(model, x, y; epoch, duration)
  println()
  println("Epoch: $(epoch), Model accuracy: $(test_accuracy(model, x, y)), Timestamp: $(duration)")
end

# Load test dataset
x_test, y_test = MLDatasets.CIFAR10.testdata()

# Log model initial test accuracy
initial_timestamp = now()
log_model_accuracy(global_model, x_test, y_test; epoch=0, duration=now() - initial_timestamp)


# Begin master node training
@profile @showprogress for ep in 1:num_epochs
  global global_model

  # Call the training from the parameter servers
  f_references = map(parameter_server_ids) do server_id
    remotecall(train_parameter_server, server_id, global_model)
  end

  server_models = map(f_references) do ref
    fetch(ref)
  end

  # Aggregate parameter servers' results
  global_model = aggregate(server_models)
  ts = now() - initial_timestamp
  log_model_accuracy(global_model, x_test, y_test; epoch=ep, duration=ts)
end
