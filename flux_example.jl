using Flux, MLDatasets
using Flux: train!, onehotbatch
using ProgressMeter

## GET DATA ##
num_epochs = 4
learning_rate = 0.01

# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()

# Load test data (images, labels)
x_test, y_test = MLDatasets.MNIST.testdata()

# Flatten images
x_train = Flux.flatten(x_train)
x_test = Flux.flatten(x_test)

# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)

# Create data loader
data_loader = Flux.DataLoader((data=x_train, label=y_train), batchsize=64, shuffle=true);

## CREATE MODEL ##
model = Chain(
    Dense(784, 64, relu),
    Dense(64, 10, relu),
    softmax
)

## DEFINE LOSS FUNCTION ##
loss_function(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

## DEFINE OPTIMIZER ##
optimizer = Flux.setup(Flux.Adam(learning_rate), model)

losses = []
@showprogress for epoch in 1:num_epochs
    println("epoch ", epoch, "/", num_epochs)
    for (x, y) in data_loader
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(optimizer, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

## VALIDATION ##
accuracy = 0
for i in 1:length(y_test)
    if (findmax(model(x_test[:, i]))[2] - 1)  == y_test[i]
        global accuracy
        accuracy = accuracy + 1
    end
end

println(accuracy / length(y_test))

