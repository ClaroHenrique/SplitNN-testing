using Flux
#using CUDA   ***

function train_client(model, data_loader)

  #model = fmap(cu, model)

  # Initializate optimizer
  optimizer = Flux.setup(Flux.Optimisers.Adam(learning_rate), model)

  Flux.train!(model, data_loader, optimizer) do m, x, y
    y_hat = m(x)
    Flux.logitcrossentropy(y_hat, y)
  end

  # Return the updated local model, i.i.e the initial
  # copy of the global model trained with local data
  model
end
