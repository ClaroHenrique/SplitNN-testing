using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
include("aggregate.jl")

# function train_parameter_server(global_model2, server_id)
#   # Call the training from the clients 
#   local_models = pmap(1:n_clients_per_server, role=:default) do i
#     client_id = (server_id-2) * n_servers + i
#     train_client(global_model, client_id)
#   end

#   # Aggregate parameter clients' results
#   server_model = aggregate(local_models)
#   server_model
# end

function train_parameter_server(global_model, server_id)
  # Call the training from the clients

  if server_id == 2
    data_ids = [(2,1),(3,2)]
  elseif server_id == 3
    data_ids = [(2,3),(3,4)]
  end

  f_references = map(data_ids) do (client_id, data_id)
    data_id = (server_id)
    remotecall(train_client, client_id, global_model, data_id)
  end

  local_models = map(f_references) do ref
    fetch(ref)
  end

  # Aggregate parameter clients' results
  server_model = aggregate(local_models)
  server_model
end
