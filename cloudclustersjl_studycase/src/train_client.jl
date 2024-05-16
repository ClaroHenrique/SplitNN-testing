using Flux

function train_client(model)
  # Initializate optimizer
  optimizer = Flux.setup(Flux.Adam(learning_rate), model)

  # Run SGD
  for (i, (x, y)) in enumerate(data_loader)
    # Calculate grads
    loss, (local_grad, ) = Flux.withgradient(model, x) do model, x
      y_hat = model(x)
      Flux.logitcrossentropy(y_hat, y)
    end

    # Use grads to update the local model
    Flux.update!(optimizer, model, local_grad)

    # Stops when reached limit of iterations per client
    if(i == iterations_per_client)
      break
    end
  end

  # Return the updated local model, i.i.e the initial
  # copy of the global model trained with local data
  model
end
