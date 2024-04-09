include("aggregate.jl")

function train_parameter_server(global_model, id)
  println("Training server $(id).")

  local_models = pmap(1:np) do i
    train_local_model(global_model)
  end

  #println("Epoch: ", ep, ", Test accuracy: ", test_accuracy(global_model))
  server_model = aggregate(local_models)
  server_model
end
