


function test_accuracy(model, x_test, y_test)
    accuracy = 0
    n = length(y_test)

    for (x,y) in Flux.DataLoader((data=x_test, label=y_test), batchsize=batch_size)
        y_true = y
        output = model(x)
        y_pred = reshape(map(x -> x[1][1] -1, findmax(output, dims=1)[2]), length(y_true))

        accuracy += sum(y_pred .== y_true)
        # println(y_pred)
    end

    accuracy / n
end

test_accuracy(model) = test_accuracy(model, x_test, y_test)

