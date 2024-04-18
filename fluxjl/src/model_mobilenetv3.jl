using MLUtils
using Flux

# Get data
x_train, y_train = MLDatasets.MNIST.traindata()
x_test, y_test = MLDatasets.MNIST.testdata()
data_loader = Flux.DataLoader((data=x_train, label=y_train), batchsize=64, partial=false)

# Define model
model = Chain(
  MLUtils.flatten,
  Dense(28*28*1 => 256, relu),            # 5_308_928 parameters
  Dense(256 => 64, relu),              # 131_328 parameters
  Dense(64 => 10),                     # 2_570 parameters
) # Total: 14 arrays, 5_702_986 parameters, 21.757 MiB.

# Train model
optimizer = Flux.setup(Flux.Adam(0.1), model)
num_batches = length(data_loader)
loss = 0.

for (i, (x, y)) in enumerate(data_loader)
    l, (local_grad, ) = Flux.withgradient(model, x) do model, x
        y_hat = model(x)
        Flux.logitcrossentropy(y_hat, y)
    end
    #println("Progress: $(i)/$(sz) or $(i/sz*100)%")
    loss += l / num_batches
    Flux.update!(optimizer, model, local_grad)
end

println("Loss: ", loss)








