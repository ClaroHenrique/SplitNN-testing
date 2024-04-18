# using Pkg; Pkg.respect_sysimage_versions(false)
# ] dev /home/henrique/Documents/teste/Distributed.jl

using ProgressMeter
using Distributed
using Profile
include("src/utils.jl")
#include("/home/henrique/Documents/teste/Distributed.jl")

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

  # Define variables
  n_servers = 2
  n_clients_per_server = 2
  n_clients = n_servers * n_clients_per_server

  main_id = 1:1
  server_ids = 2:1+n_servers
  client_ids = 1+n_servers+1:1+n_servers+1+n_clients

  main_worker_pool = WorkerPool(main_id)
  server_worker_pool = WorkerPool(server_ids)
  client_worker_pool = WorkerPool(client_ids)

  if myid() == 1
    println("WORKERS POOLS IDS")
    println("main_id: $(main_id)")
    println("server_ids: $(server_ids)")
    println("client_ids: $(client_ids)")
  end

  num_epochs = 3
  learning_rate = 0.001
  batch_size = 64
  batchs_per_epoch = 100
  random_seed = 42

  #define train functions
  include("src/train_parameter_server.jl")
  include("src/train_client.jl")
end

println("Defining model")
include("src/model_custom.jl")
global_model = custom_model

println("Testing model")
println("Epoch: $(0), Accuracy: $(test_accuracy(global_model))")


@profile @showprogress for ep in 1:num_epochs
  global global_model

  server_models = pmap(server_worker_pool, 1:n_servers, role=:default) do i
    train_parameter_server(global_model, i)
  end

  global_model = aggregate(server_models)
  println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
end

