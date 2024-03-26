
function test_accuracy(model, x_test, y_test)
    accuracy = 0
    n = length(y_test)
    output = model(x_test)

    for i in 1:n
        y_true = y_test[i]
        y_pred = findmax(output[:, i])[2][1] - 1
        if (y_pred == y_true)
            accuracy = accuracy + 1
        end
    end

    accuracy / n
end
