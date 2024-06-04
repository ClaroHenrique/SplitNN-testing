using Flux
using MLUtils

model = Chain(
  Dense(10 => 64, relu),
  Dropout(0.5),
  Dense(64 => 1, relu),
)               

println(model(rand(10, 1)))

