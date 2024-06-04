using MLUtils
using Flux

using Metalhead
mobilenetv3_small = MobileNetv3(:small, nclasses=10) # 5.938 MiB
mobilenetv3_large = MobileNetv3(:large, nclasses=10) # 16.247 MiB

