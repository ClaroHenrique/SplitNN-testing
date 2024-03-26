using Flux, MLDatasets
using Flux: train!, onehotbatch
using ProgressMeter

## GET DATA ##
num_epochs = 3
learning_rate = 0.01
batchsize = 64

# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()
# Load test data (images, labels)
x_test, l = MLDatasets.MNIST.testdata()

# Flatten images
x_train = Flux.flatten(x_train)
x_test = Flux.flatten(x_test)

# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)

# Create data loader
data_loader = Flux.DataLoader((data=x_train, label=y_train), batchsize=batchsize, shuffle=true, partial=false);

function train_local_model(model)
  x, y = first(data_loader)
  optimizer = Flux.setup(Flux.Adam(learning_rate), model)
  num_batches = 10
  loss = 0.

  for (x, y) in first(data_loader, num_batches)
    l, (local_grad, ) = Flux.withgradient(model, x) do model, x
      y_hat = model(x)
      Flux.logitcrossentropy(y_hat, y)
    end

    loss += l / num_batches
    Flux.update!(optimizer, model, local_grad)
  end
  
  println(loss)
  model
end


function aggregate(models::AbstractVector{T}) where T <: Chain
  new_model_layers = []
  n_models = length(models)

  # iterate over layers of the model
  for (i, layer) in enumerate(model)
    if layer isa Dense
      params = Flux.params(layer)
      new_layer_w = zeros(Float32, size(params[1]))
      new_layer_b = zeros(Float32, size(params[2]))

      # sum current layer's params from all the models 
      for m in models
        params = Flux.params(m[i])
        new_layer_w += params[1]
        new_layer_b += params[2]
      end

      # get mean from the division
      new_layer_w ./= n_models
      new_layer_b ./= n_models
      
      push!(new_model_layers, Dense(new_layer_w, new_layer_b, layer.σ))
    else
      push!(new_model_layers, layer)
    end
  end

  Chain(new_model_layers...)
end


model = Chain(
    Dense(784, 32, relu),
    Dense(32, 10),
    softmax
)

m1 = train_local_model(deepcopy(model))
m2 = train_local_model(deepcopy(model))
m3 = train_local_model(deepcopy(model))
models = [m1, m2, m3]

new_model = aggregate(models)
new_model

# Flux.params(new_model[1])[1][1,1] ≈ sum([Flux.params(m1[1])[1][1,1], Flux.params(m2[1])[1][1,1], Flux.params(m3[1])[1][1,1]])/n_models


