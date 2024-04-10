include("aggregate.jl")

function train_parameter_server(global_model, server_id)
  println("Training server $(server_id) (Process: $(myid())).")

  local_models = pmap(client_worker_pool, 1:n_clients_per_server, role=:default) do i
    client_id = (server_id-1) * n_servers + i
    train_client(global_model, client_id)
  end

  #println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
  server_model = aggregate(local_models)
  server_model
end
