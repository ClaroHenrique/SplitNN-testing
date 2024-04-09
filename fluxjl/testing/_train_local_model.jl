using Flux

function train_local_model(model, data_loader, learning_rate)
  optimizer = Flux.setup(Flux.Adam(learning_rate), model)
  num_batches = length(data_loader)
  loss = 0.

  for (x, y) in data_loader
    l, (local_grad, ) = Flux.withgradient(model, x) do model, x
      y_hat = model(x)
      Flux.logitcrossentropy(y_hat, y)
    end

    loss += l / num_batches
    Flux.update!(optimizer, model, local_grad)
  end
  
  println("Loss: ", loss)
  model
end
