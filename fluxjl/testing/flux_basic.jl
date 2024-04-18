using MLUtils
using Flux

# Get data
x_train, y_train = MLDatasets.CIFAR10.traindata()
x_test, y_test = MLDatasets.CIFAR10.testdata()
data_loader = Flux.DataLoader((data=x_train, label=y_train), batchsize=64, partial=false, shuffle=true)

# Define model
model = Chain(
  Conv((3, 3), 3 => 64, relu, pad=1),  # 1_792 parameters
  Conv((3, 3), 64 => 64, relu, pad=1),  # 36_928 parameters
  MaxPool((2, 2), pad=1),
  MLUtils.flatten,
  Dense(18496 => 256, relu),            # 5_308_928 parameters
  Dense(256 => 64, relu),              # 131_328 parameters
  Dense(64 => 10),                     # 2_570 parameters
  softmax
) # Total: 14 arrays, 5_702_986 parameters, 21.757 MiB.

model(x_train[:,:,:,1:1])

# Train model
optimizer = Flux.setup(Flux.Adam(0.001), model)
num_batches = length(data_loader)
loss = 0.

for (i, (x, y)) in enumerate(data_loader)
    println("$(i)/$(length(data_loader))")
    l, (local_grad, ) = Flux.withgradient(model, x) do model, x
        y_hat = model(x)
        Flux.logitcrossentropy(y_hat, Flux.onehotbatch(y, 0:9))
    end

    loss += l / num_batches
    Flux.update!(optimizer, model, local_grad)
end

println("Loss: ", loss)

function test_accuracy(model, x_test, y_test)
  accuracy = 0
  n = length(y_test)

  for (x,y) in Flux.DataLoader((data=x_test, label=y_test), batchsize=batch_size)
      y_true = y
      output = model(x)
      y_pred = reshape(map(x -> x[1][1] -1, findmax(output, dims=1)[2]), length(y_true))

      accuracy += sum(y_pred .== y_true)
      println(y_pred)
  end
  accuracy / n
end

test_accuracy(model, x_test, y_test)
# 0.3185

model(x_test)




