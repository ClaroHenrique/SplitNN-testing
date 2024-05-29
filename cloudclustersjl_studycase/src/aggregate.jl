using Flux

function aggregate(models::AbstractVector{T}) where T <: Chain
  new_model_layers = []
  n_models = length(models)

  # iterate over layers of the model
  for (i, layer) in enumerate(models[1])
    if layer isa Dense || layer isa Conv
      params = Flux.params(layer)
      new_layer_w = CUDA.zeros(size(params[1])) #zeros(Float32, size(params[1])) 
      new_layer_b = CUDA.zeros(size(params[2])) #zeros(Float32, size(params[2]))

      # sum current layer's params from all the models 
      for m in models
        params = Flux.params(m[i])
        params_1 = cu(params[1])
        params_2 = cu(params[2])
        new_layer_w += params_1 
        new_layer_b += params_2 
      end

      # get mean from the division
      new_layer_w ./= n_models
      new_layer_b ./= n_models
      
      new_layer_w = new_layer_w |> cpu
      new_layer_b = new_layer_b |> cpu

      if layer isa Dense
        push!(new_model_layers, Dense(new_layer_w, new_layer_b, layer.σ))
      else # layer isa Conv
        new_layer = Conv(new_layer_w, new_layer_b, layer.σ;
          stride=layer.stride,
          pad=layer.pad,
          dilation=layer.dilation,
          groups=layer.groups
        )

        push!(new_model_layers, new_layer)
      end
    else
      push!(new_model_layers, layer)
    end
  end

  Chain(new_model_layers...)
end


