using MLUtils
using Flux

vgg16 = Chain(
  Conv((3, 3), 3 => 64, relu, pad=1),  # 1_792 parameters
  Conv((3, 3), 64 => 64, relu, pad=1),  # 36_928 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 64 => 128, relu, pad=1),  # 73_856 parameters
  Conv((3, 3), 128 => 128, relu, pad=1),  # 147_584 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 128 => 256, relu, pad=1),  # 295_168 parameters
  Conv((3, 3), 256 => 256, relu, pad=1),  # 590_080 parameters
  Conv((3, 3), 256 => 256, relu, pad=1),  # 590_080 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 256 => 512, relu, pad=1),  # 1_180_160 parameters
  Conv((3, 3), 512 => 512, relu, pad=1),  # 2_359_808 parameters
  Conv((3, 3), 512 => 512, relu, pad=1),  # 2_359_808 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 512 => 512, relu, pad=1),  # 2_359_808 parameters
  Conv((3, 3), 512 => 512, relu, pad=1),  # 2_359_808 parameters
  Conv((3, 3), 512 => 512, relu, pad=1),  # 2_359_808 parameters
  MaxPool((2, 2)),
  MLUtils.flatten,
  Dense(25088 => 4096, relu),       # 102_764_544 parameters
  Dropout(0.5),
  Dense(4096 => 4096, relu),        # 16_781_312 parameters
  Dropout(0.5),
  Dense(4096 => 10),                # 40_970 parameters
)

