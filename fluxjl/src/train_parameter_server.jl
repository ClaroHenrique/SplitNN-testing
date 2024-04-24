using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
include("aggregate.jl")

function train_parameter_server(global_model)
  # Call the training from the clients
  f_references = map(client_ids) do client_id
    remotecall(train_client, client_id, global_model)
  end

  local_models = map(f_references) do ref
    fetch(ref)
  end

  # Aggregate parameter clients' results
  server_model = aggregate(local_models)
  server_model
end
