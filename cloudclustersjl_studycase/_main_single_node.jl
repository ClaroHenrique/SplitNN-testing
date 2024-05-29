using ProgressMeter
using Profile
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates
using CUDA

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")

### Inicitalizate client nodes ###

println("Initializating clients")

# Define constants
learning_rate = 0.001
batch_size = 32
iterations_per_client = 100
data_loader = dataset_loader(batch_size)

# Define model
model = fmap(cu,custom_model)
#model = custom_model

# include("src/models/vgg16.jl")
# model = vgg16

# Log model initial test accuracy
initial_timestamp = now()
log_test_accuracy(model; epoch=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_epochs = 20

@profile @showprogress for ep in 1:num_epochs
  global model
  train_client(model)
  log_test_accuracy(model; epoch=ep, timestamp=now()-initial_timestamp)
end
