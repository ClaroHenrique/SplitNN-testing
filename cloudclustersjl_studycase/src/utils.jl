function test_accuracy(model, x_test, y_test; batchsize=32)
    accuracy = 0
    n = length(y_test)

    for (x,y) in Flux.DataLoader((data=x_test, label=y_test), batchsize=batchsize)
        y_true = y
        output = model(x)
        y_pred = reshape(map(x -> x[1][1] -1, findmax(output, dims=1)[2]), length(y_true))

        accuracy += sum(y_pred .== y_true)
    end

    accuracy / n
end

# Load test dataset
x_test, y_test = CIFAR10(split=:test)[:]  # MLDatasets.CIFAR10.testdata()

#x_test = cu(x_test)
#y_test = cu(y_test)

function log_model_accuracy(model, x, y; epoch, timestamp)
  println()
  println("Epoch: $(epoch), Model accuracy: $(test_accuracy(model, x, y)), Timestamp: $(timestamp)")
end

log_test_accuracy(model; epoch, timestamp) = log_model_accuracy(model, x_test, y_test; epoch=epoch, timestamp=timestamp)


