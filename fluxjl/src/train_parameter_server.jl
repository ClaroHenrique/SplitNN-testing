include("aggregate.jl")

function train_parameter_server(global_model, server_id)
  # Call the training from the clients 
  local_models = pmap(1:n_clients_per_server, role=:default) do i
    client_id = (server_id-1) * n_servers + i
    train_client(global_model, client_id)
  end

  # Aggregate parameter clients' results
  server_model = aggregate(local_models)
  server_model
end
