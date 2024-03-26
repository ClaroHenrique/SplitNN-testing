using ProgressMeter
using Distributed
include("utils.jl")

# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
  # load dependencies
  using Flux, MLDatasets
  using Flux: train!, onehotbatch
  include("fed_mnist.jl")
  include("train_local_model.jl")
  include("aggregate.jl")

  # Define variables
  np = 4
  id = (myid() % np) + 1
  num_epochs = 3
  learning_rate = 0.01
  batch_size = 64
  random_seed = 42
  println("id: ", id, ", p: ", np)

  # define data loader
  data_loader = fed_mnist_loader(id=id, np=np, batch_size=batch_size, random_seed=random_seed)

  # define train function 
  train_local_model(model) = train_local_model(model, data_loader, learning_rate)

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

epochs = 3
@showprogress for ep in 1:epochs
  global global_model

  local_models = pmap(1:np) do i
    train_local_model(global_model)
  end

  global_model = aggregate(local_models)
  println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
end
