using ProgressMeter
using Distributed
using Profile
include("src/utils.jl")

# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
  # Load dependencies
  using Flux, MLDatasets
  using Flux: train!, onehotbatch
  include("src/dataset_loader.jl")
  include("src/aggregate.jl")
  include("src/train_parameter_server.jl")
  include("src/train_client.jl")

  # Define global constants
  n_servers = 2
  n_clients_per_server = 2
  n_clients = n_servers * n_clients_per_server

  num_epochs = 20
  learning_rate = 0.001
  batch_size = 32
  batchs_per_epoch = 100
  random_seed = 42

  if myid() == 1
    # Load test dataset
    x_test, y_test = MLDatasets.CIFAR10.testdata()

    # Create federated train dataset file, if it is not present  
    try
        JLD2.load("dataset/train_data_$(n_clients)_b$(batch_size).jld2", "ok")
    catch err
        if isa(err, ArgumentError)
            initializate_fed_dataset(n_clients=n_clients, batch_size=batch_size)
        else
            throw(e)
        end
    end
  end
end


# Define model
include("src/models/custom.jl")
global_model = custom_model

function log_model_accuracy(model, x, y; epoch)
  println("Epoch: $(epoch), Model accuracy: $(test_accuracy(model, x, y))")
end

# Log model initial test accuracy
log_model_accuracy(global_model, x_test, y_test; epoch=0)

# Begin master node training
@profile @showprogress for ep in 1:num_epochs
  global global_model

  # Call the training from the parameter servers 
  server_models = pmap(1:n_servers, role=:default) do i
    train_parameter_server(global_model, i)
  end

  # Aggregate parameter servers' results
  global_model = aggregate(server_models)
  log_model_accuracy(global_model, x_test, y_test; epoch=ep)
end
