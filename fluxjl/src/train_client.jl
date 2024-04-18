include("aggregate.jl")
using Flux

function train_client(model, client_id)
  println("Client $(client_id) (Process(m, w): $(myid(role=:master)), $(myid(role=:worker))) is training")
  data_loader = dataset_loader(id=client_id, np=n_clients, batch_size=batch_size, random_seed=random_seed)
  sz = length(data_loader)

  optimizer = Flux.setup(Flux.Adam(learning_rate), model)
  num_batches = length(data_loader)
  loss = 0.

  for (i, (x, y)) in enumerate(data_loader)
    l, (local_grad, ) = Flux.withgradient(model, x) do model, x
      y_hat = model(x)
      Flux.logitcrossentropy(y_hat, y)
    end
    # println("Progress: $(i)/$(sz) or $(i/sz*100)%")
    loss += l / num_batches
    Flux.update!(optimizer, model, local_grad)
    if(i == batchs_per_epoch)
      break
    end
  end

  # println("Loss: ", loss)
  model
end
