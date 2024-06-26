using CUDA

function test_accuracy(model, data_loader)
    correct = 0
    total = 0
    for (x, y) in data_loader
        n = size(x)[end]
        output = model(x) |> cpu
        y_pred = reshape(map(x -> x[1][1] -1, findmax(output  , dims=1)[2]), n)
        y_true = reshape(map(x -> x[1][1] -1, findmax(y |> cpu, dims=1)[2]), n)
        correct += sum(y_pred .== y_true)
        total += n
    end
    correct / total
end

function log_model_accuracy(model, data_loader; iteration, timestamp)
  println()
  println("Iteration: $(iteration), Model accuracy: $(test_accuracy(model, data_loader)), Timestamp: $(timestamp)")
end
