using Flux, MLDatasets

function dataset_loader(batch_size)
  # Load training data (images, labels)
  x_train, y_train = MLDatasets.CIFAR10.traindata()

  # Convert labels to one-hot encode
  y_train = Flux.onehotbatch(y_train, 0:9)

  # Create data loader
  Flux.DataLoader((data=x_train, label=y_train), batchsize=batch_size, partial=false, shuffle=true);
end
