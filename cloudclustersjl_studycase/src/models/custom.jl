using Flux
using MLUtils

custom_model = Chain(
  Conv((3, 3), 3 => 32, relu, pad=1),   # 896 parameters
  Conv((3, 3), 32 => 32, relu, pad=1),  # 9_248 parameters
  MaxPool((2, 2), pad=1),
  Conv((3, 3), 32 => 64, relu, pad=1),  # 18_496 parameters
  Conv((3, 3), 64 => 64, relu, pad=1),  # 36_928 parameters
  MaxPool((2, 2), pad=1),
  MLUtils.flatten,
  Dense(5184 => 512, relu),             # 2_654_720 parameters
  #Dropout(0.5),
  Dense(512 => 256, relu),              # 131_328 parameters
  #Dropout(0.5),
  Dense(256 => 10),                     # 2_570 parameters
)                   
# Total: 14 arrays, 2_854_186 parameters, 10.890 MiB.

