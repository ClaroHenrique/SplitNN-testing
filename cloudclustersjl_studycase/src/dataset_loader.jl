using Flux, MLDatasets, Images

function dataset_loader(data; batch_size, img_dims, n_batches)
  # Load training data (images, labels)
  x_train, y_train = data

  # Resize images
  fx(x) = resize_images(x, (img_dims)) #|> cu

  # Convert labels to one-hot encode
  fy(y) = Flux.onehotbatch(y, 0:9) #|> cu
 
  # Create data loader
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

function resize_images(data, new_dims)
  n = size(data)[end]
  num_channels = size(data)[end-1]
  new_data = zeros(Float32, new_dims..., num_channels, n)
  for i in 1:n
    new_data[:,:,:,i] .= imresize(data[:,:,:,i], new_dims)
  end
  new_data
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
