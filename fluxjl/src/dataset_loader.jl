using Flux, MLDatasets
using Images
using Random

function preprocess_imgs(x)
  d = size(x)
  # add channel dim
  x = reshape(x, tuple(d[1], d[2], channels, last(d)))
  mapslices(x, dims=(1,2)) do img
    # resize images
    resized_img = imresize(img, input_dim)
    resized_img
  end
end

## MNIST DATA ##
# x_train, y_train = MLDatasets.MNIST.traindata()
# x_test, y_test = MLDatasets.MNIST.testdata()
# input_dim = (32, 32)
# channels = 1
# x_train = preprocess_imgs(x_train)
# x_test = preprocess_imgs(x_test)

# ## CIFAR10 DATA ##
x_train, y_train = MLDatasets.CIFAR10.traindata()
x_test, y_test = MLDatasets.CIFAR10.testdata()
input_dim = (32, 32)
channels = 3



#x_train = preprocess_imgs(x_train)
#x_test = preprocess_imgs(x_test)


function dataset_loader(;id=1, np=4, batch_size=64, random_seed=42)
  # Load training data (images, labels)
  
  n = (last(size(x_train)) ÷ batch_size) * batch_size
  p_size = n ÷ np

  Random.seed!(random_seed)
  index = randperm(n)
  local_x_train = x_train[:, :, :, index]
  local_y_train = y_train[index]

  i = (id-1)*p_size + 1
  j = i+p_size-1
  local_x_train = local_x_train[:, :, :, i:j]
  local_y_train = local_y_train[i:j]

  # Create labels batch
  local_y_train = Flux.onehotbatch(local_y_train, 0:9)

  # Create data loader
  Flux.DataLoader((data=local_x_train, label=local_y_train), batchsize=batch_size, partial=false, shuffle=true);
end


