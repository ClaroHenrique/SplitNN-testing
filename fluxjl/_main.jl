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
  n_client_per_server = 2   
  id = (myid() % np) + 1
  num_epochs = 3
  learning_rate = 0.01
  batch_size = 64
  random_seed = 42
  println("id: ", id, ", p: ", np)

  # define data loader
  data_loader = fed_mnist_loader(id=id, np=np, batch_size=batch_size, random_seed=random_seed)

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
    Dense(64, 10),
    softmax
)
    
println("Epoch: ", 0, ", Accuracy: ", test_accuracy(global_model))

function pmap_mock(f::Function, ids)
  results = []
  for id in ids
      push!(results, f(id))
  end
  results
end

epochs = 3
@showprogress for ep in 1:epochs
  global global_model

  server_models = pmap_mock(1:2) do i
    train_parameter_server(global_model, i)
  end

  global_model = aggregate(server_models)
  println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
end
