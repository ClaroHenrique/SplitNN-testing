using Flux, MLDatasets
using Images
using Random
import JLD2

function initializate_fed_dataset(;n_clients, batch_size, random_seed=42)
  println("Creating federated dataset file")

  # Check if already exists
  try
    JLD2.load("dataset/train_data_$(n_clients)_b$(batch_size).jld2", "ok")
  catch err
      if isa(err, ArgumentError)
          return
      else
          throw(e)
      end
  end

  # Import CIFAR10 train dataset
  x_train, y_train = MLDatasets.CIFAR10.traindata()
  
  # Calculate new size to each client
  n = (last(size(x_train)) ÷ batch_size) * batch_size
  p_size = n ÷ n_clients

  # Suffle dataset 
  Random.seed!(random_seed)
  index = randperm(n)
  x_train = x_train[:, :, :, index]
  y_train = y_train[index]

  # Save each partition of data in a dict
  fed_dataset = Dict()
  for id in 1:n_clients
    i = (id-1)*p_size + 1
    j = i+p_size-1
    fed_dataset["data_train_$(id)"] = x_train[:, :, :, i:j], y_train[i:j]
  end
  fed_dataset["ok"] = true

  # Save in file
  JLD2.save("dataset/train_data_$(n_clients)_b$(batch_size).jld2", fed_dataset)
end


function dataset_loader(;id, n_clients, batch_size)
  # Load training data (images, labels)
  
  x_train, y_train = JLD2.load("dataset/train_data_$(n_clients)_b$(batch_size).jld2", "data_train_$(id)")
  # input_dim = (32, 32)

  # Create labels batch
  y_train = Flux.onehotbatch(y_train, 0:9)

  # Create data loader
  Flux.DataLoader((data=x_train, label=y_train), batchsize=batch_size, partial=false, shuffle=true);
end


