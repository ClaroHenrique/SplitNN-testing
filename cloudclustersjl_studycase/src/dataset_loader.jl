using Flux, MLDatasets, Images
using CUDA


function dataset_loader(data; img_dims, batch_size=32, n_batches=2^50)
  # Load training data (images, labels)
  x_train, y_train = data

  # Resize images
  fx(x) = resize_images(x, img_dims) |> gpu #TODO: resize in GPU?

  # Convert labels to one-hot encode
  fy(y) = Flux.onehotbatch(y, 0:9) |> gpu
 
  # Create data loader
  # Flux.DataLoader((x_train, y_train) |> gpu, batchsize=batch_size, partial=false, shuffle=true)

  DataLoaderTransform(
    data=x_train,
    label=y_train,
    n_batches=n_batches,
    fx=fx,
    fy=fy,
    batch_size=batch_size,
    partial=false,
    shuffle=true
  );
end

function resize_images(imgs, new_dims)
  n = size(imgs)[end]
  num_channels = size(imgs)[end-1]
  new_imgs = zeros(Float32, new_dims..., num_channels, n) #new_imgs = CUDA.zeros(Float32, new_dims..., num_channels, n)
  for i in 1:n
    new_imgs[:,:,:,i] .= imresize(imgs[:,:,:,i], new_dims)
  end
  new_imgs
end

# Iterate over data with lazy transform #
struct DataLoaderTransform
  data_loader::Flux.DataLoader
  n_batches::Int
  fx::Function
  fy::Function

  function DataLoaderTransform(;data, label, n_batches, fx, fy, batch_size, partial, shuffle)
    data_loader = Flux.DataLoader(
      (data=data, label=label),
      batchsize=batch_size,
      partial=partial,
      shuffle=shuffle,
    )
    new(data_loader, n_batches, fx, fy)
  end
end 

function Base.iterate(dlt::DataLoaderTransform)
  next = iterate(dlt.data_loader)
  if isnothing(next) || next[2][2] > dlt.n_batches
    return nothing
  end
  ((x, y), state) = next
  ((dlt.fx(x), dlt.fy(y)), state)
end

function Base.iterate(dlt::DataLoaderTransform, state)
  next = iterate(dlt.data_loader, state)
  if isnothing(next) || next[2][2] > dlt.n_batches
    return nothing
  end

  ((x, y), state) = iterate(dlt.data_loader, state)
  ((dlt.fx(x), dlt.fy(y)), state)
end
