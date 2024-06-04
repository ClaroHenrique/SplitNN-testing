using Flux

function aggregate(models)
  _, rebuild = Flux.destructure(models[1])
  ws = map(m -> Flux.destructure(m)[1], models)

  new_w = sum(ws) ./ length(models)
  rebuild(new_w)
end
