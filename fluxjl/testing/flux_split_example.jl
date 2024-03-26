using Flux, MLDatasets
using Flux: train!, onehotbatch
import ChainRules: rrule
using ChainRulesCore
using ProgressMeter

## GET DATA ##
num_epochs = 3
learning_rate = 0.01
    
# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()
# Load test data (images, labels)
x_test, l = MLDatasets.MNIST.testdata()

# Flatten images
x_train = Flux.flatten(x_train)
x_test = Flux.flatten(x_test)

# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)

# Create data loader
data_loader = Flux.DataLoader((data=x_train, label=y_train), batchsize=64, shuffle=true, partial=false);

global_cutlayerout = Nothing
global_cutlayergrad = Nothing

function cutlayerfunction(x)
    global global_cutlayerout
    global_cutlayerout = x
    42 # just a placeholder
end

function cutlayergrad()
    global global_cutlayergrad
    global_cutlayergrad
end

function rrule(::typeof(cutlayerfunction), x)
    customlayer_pullback(Δy) = (NoTangent(), cutlayergrad())
    return cutlayerfunction(x), customlayer_pullback
end

## CREATE MODEL ##
clientmodel = Chain(
    Dense(784, 64, relu),
    Dense(64, 32, relu),
    cutlayerfunction
)

servermodel = Chain(
    Dense(32, 64, relu),
    Dense(64, 10),
    softmax
)


## VALIDATION ##
function test_accuracy()
    accuracy = 0
    nt = length(y_test)

    clientmodel(x_test)
    output = servermodel(global_cutlayerout)
    
    for i in 1:nt
        y_true = y_test[i]
        y_pred = findmax(output[:, i])[2][1] - 1
        if (y_pred == y_true)
            accuracy = accuracy + 1
        end
    end
    accuracy / nt
end

## DEFINE OPTIMIZER ##
serveroptimizer = Flux.setup(Flux.Adam(learning_rate), servermodel)
clientoptimizer = Flux.setup(Flux.Adam(learning_rate), clientmodel)


losses = []
for epoch in 1:num_epochs
    println("="^40)
    println("Epoch ", epoch, "/", num_epochs)
    sum_loss = 0
    @showprogress desc="Mini batch gradient descent." for (x, y) in data_loader
        # client
        clientmodel(x) # assign value to global_cutlayerout (by side effect)
        
        # server
        server_input = copy(global_cutlayerout)
        loss, (servermodelgrad, serverinputgrad) = Flux.withgradient(servermodel, server_input) do model, x
            y_hat = model(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        sum_loss += loss

        # client
        global global_cutlayergrad = copy(serverinputgrad)
        clientmodelgrad, = Flux.gradient(clientmodel) do model
            model(x)
        end
        
        # client
        Flux.update!(clientoptimizer, clientmodel, clientmodelgrad)

        # server
        Flux.update!(serveroptimizer, servermodel, servermodelgrad)
        push!(losses, loss)  # logging, outside gradient context
    end
    println("loss: ", sum_loss / length(data_loader))
    println("test accuracy: ", test_accuracy())

end
println("final test accuracy: ", test_accuracy())



