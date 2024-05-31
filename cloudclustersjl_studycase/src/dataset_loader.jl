using Flux, MLDatasets

function dataset_loader(batch_size)
  # Load training data (images, labels)
  x_train, y_train = CIFAR10(split=:train)[:] # MLDatasets.CIFAR10.traindata() 

  #x_train = cu(x_train)   ***

  # Convert labels to one-hot encode
  y_train = Flux.onehotbatch(y_train, 0:9)

  #y_train = cu(y_train)   ***
 
  # Create data loader
  Flux.DataLoader((x_train, y_train) |> gpu, batchsize=batch_size, partial=false, shuffle=true);
end
