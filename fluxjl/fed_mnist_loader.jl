using Flux, MLDatasets
using Random

## GET DATA ##
x_train, y_train = MLDatasets.MNIST.traindata()

function fed_mnist_loader(;id=1, np=4, batch_size=64, random_seed=42)
  # Load training data (images, labels)
  
  n = (size(x_train)[3] ÷ batch_size) * batch_size
  p_size = n ÷ np

  Random.seed!(random_seed)
  index = randperm(n)
  local_x_train = x_train[:, :, index]
  local_y_train = y_train[index]

  i = (id-1)*p_size + 1
  j = i+p_size-1
  local_x_train = local_x_train[:, :, i:j]
  local_y_train = local_y_train[i:j]

  # Flatten images
  local_x_train = Flux.flatten(local_x_train)

  # Create labels batch
  local_y_train = Flux.onehotbatch(local_y_train, 0:9)

  # Create data loader
  Flux.DataLoader((data=local_x_train, label=local_y_train), batchsize=batch_size, partial=false);
end


