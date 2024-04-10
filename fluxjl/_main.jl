# using Pkg; Pkg.respect_sysimage_versions(false)
# ] dev /home/henrique/Documents/teste/Distributed.jl

using ProgressMeter
using Distributed
include("utils.jl")
#include("/home/henrique/Documents/teste/Distributed.jl")

# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
  # load dependencies
  using Flux, MLDatasets
  using Flux: train!, onehotbatch
  include("fed_mnist_loader.jl")
  include("aggregate.jl")

  # Define variables
  n_servers = 2
  n_clients_per_server = 2
  n_clients = n_servers * n_clients_per_server

  main_id = 1:1
  server_ids = 2:1+n_servers
  client_ids = 1+n_servers+1:1+n_servers+1+n_clients
  
  if myid() == 1
    println("WORKERS POOLS IDS")
    println("main_id: $(main_id)")
    println("server_ids: $(server_ids)")
    println("client_ids: $(client_ids)")
  end

  main_worker_pool = WorkerPool(main_id)
  server_worker_pool = WorkerPool(server_ids)
  client_worker_pool = WorkerPool(client_ids)

  num_epochs = 3
  learning_rate = 0.01
  batch_size = 64
  random_seed = 42

  #define train functions
  include("train_parameter_server.jl")
  include("train_client.jl")

  # define test accuracy function
  x_test, y_test = MLDatasets.MNIST.testdata()
  x_test = Flux.flatten(x_test)
  test_accuracy(model) = test_accuracy(model, x_test, y_test)
end


global_model = Chain(
    Dense(784, 64, relu),
    Dense(64, 128, relu),
    Dense(128, 32, relu),
    Dense(32, 64, relu),
    Dense(64, 10),
    softmax
)

# using Metalhead
# global_model = resnet(18; imsize=(28, 28))
    
println("Epoch: ", 0, ", Accuracy: ", test_accuracy(global_model))

@showprogress for ep in 1:num_epochs
  global global_model

  server_models = pmap(server_worker_pool, 1:n_servers, role=:default) do i
    train_parameter_server(global_model, i)
  end

  global_model = aggregate(server_models)
  println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
end
