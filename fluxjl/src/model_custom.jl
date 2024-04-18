using MLUtils
using Flux

custom_model = Chain(
  Conv((3, 3), 3 => 32, relu, pad=1),   # 1_792 parameters
  Conv((3, 3), 32 => 32, relu, pad=1),  # 36_928 parameters
  MaxPool((2, 2), pad=1),
  Conv((3, 3), 32 => 64, relu, pad=1),  # 73_856 parameters
  Conv((3, 3), 64 => 64, relu, pad=1),  # 147_584 parameters
  MaxPool((2, 2), pad=1),
  MLUtils.flatten,
  Dense(5184 => 512, relu),            # 5_308_928 parameters
  Dropout(0.5),
  Dense(512 => 256, relu),              # 131_328 parameters
  Dropout(0.5),
  Dense(256 => 10),                     # 2_570 parameters
) # Total: 14 arrays, 5_702_986 parameters, 21.757 MiB.

