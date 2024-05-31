using Flux
#using CUDA   ***

function train_client(model, data_loader)

  model = model |> gpu

  # Initializate optimizer
  optimizer = Flux.setup(Flux.Optimisers.Adam(learning_rate), model)

  # Run SGD


  i=0  
  for (x, y) in data_loader
    # Calculate grads
    loss, (local_grad, ) = Flux.withgradient(model, x) do model, x
      y_hat = model(x)
      Flux.logitcrossentropy(y_hat, y)
    end # |> gpu

    # Use grads to update the local model
    Flux.update!(optimizer, model, local_grad) # |> gpu

    # Stops when reached limit of iterations per client
  #  if(i == iterations_per_client)
  #    break
  #  end
    i+=1
  end # |> gpu   ***
  
  @info "$i iterations"
#=  
=======
>>>>>>> 1d30a67e914e064bea32f3aedeb2fdbd2f894295
  Flux.train!(model, data_loader, optimizer) do m, x, y
    y_hat = m(x)
    Flux.logitcrossentropy(y_hat, y)
  end
=#

  # Return the updated local model, i.i.e the initial
  # copy of the global model trained with local data
  model # |> cpu   ***
end
