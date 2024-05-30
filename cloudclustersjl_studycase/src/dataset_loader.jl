using Flux, MLDatasets, Images
include("utils.jl")

function dataset_loader(batch_size, img_dims)
  # Load training data (images, labels)
  x_train, y_train = MLDatasets.CIFAR10.traindata()

  # Resize image
  x_transform(x) = resize_images(x, img_dims)

  # Convert labels to one-hot encode
  y_transform(y) = Flux.onehotbatch(y, 0:9)

  # Create data loader
  # Flux.DataLoader((data=x_train, label=y_train), batchsize=batch_size, partial=false, shuffle=true);
  DataLoaderTransform(data=x_train, label=y_train,
    fx= x_transform,
    fy= y_transform,
    batch_size= batch_size,
    partial= false,
    shuffle= true,
  );
end

# Data transformers #
struct DataLoaderTransform
  data_loader::Any
  fx::Function
  fy::Function

  function DataLoaderTransform(;data, label, fx, fy, batch_size, partial, shuffle)
    data_loader = Flux.DataLoader(
      (data=data, label=label),
      batchsize=batch_size,
      partial=partial,
      shuffle=shuffle,
    )
    new(data_loader, fx, fy)
  end
end 

function Base.iterate(dlt::DataLoaderTransform)
  ((x, y), state) = iterate(dlt.data_loader)
  ((dlt.fx(x), dlt.fy(y)), state)
end

function Base.iterate(dlt::DataLoaderTransform, state)
  ((x, y), state) = iterate(dlt.data_loader, state)
  ((dlt.fx(x), dlt.fy(y)), state)
end

img= x_train[:,:,:,64]
img2 = zeros(3,32,32)
img2[1,:,:] = img[:,:,1]
img2[2,:,:] = img[:,:,2]
img2[3,:,:] = img[:,:,3]
colorview(RGB, img2)

img= x_train[:,:,:,64]
imgg = resize_images(reshape(img, (32,32,3,1)), (224,224))
img3 = zeros(3,224,224)
img3[1,:,:] = imgg[:,:,1,1]
img3[2,:,:] = imgg[:,:,2,1]
img3[3,:,:] = imgg[:,:,3,1]
colorview(RGB, img3)

colorview(RGB, img2)
colorview(RGB, img3)

save("2.png", colorview(RGB, img2))
save("3.png", colorview(RGB, img3))