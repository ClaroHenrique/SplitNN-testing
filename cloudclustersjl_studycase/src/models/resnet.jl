using MLUtils
using Flux

using Metalhead
resnet18 = ResNet(18, nclasses=10) # 42.717 MiB
