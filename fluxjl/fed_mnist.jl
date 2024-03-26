using Flux, MLDatasets
using Random

## GET DATA ##

function fed_mnist_loader(;id=1, np=4, batch_size=64, random_seed=42)
  # Load training data (images, labels)
  x_train, y_train = MLDatasets.MNIST.traindata()
  n = (size(x_train)[3] ÷ batch_size) * batch_size
  p_size = n ÷ np

  Random.seed!(random_seed)
  index = randperm(n)
  x_train = x_train[:, :, index]
  y_train = y_train[index]

  i = (id-1)*p_size + 1
  j = i+p_size-1
  x_train = x_train[:, :, i:j]
  y_train = y_train[i:j]

  # Flatten images
  x_train = Flux.flatten(x_train)

  # Create labels batch
  y_train = Flux.onehotbatch(y_train, 0:9)

  # Create data loader
  Flux.DataLoader((data=x_train, label=y_train), batchsize=batch_size, partial=false);
end


