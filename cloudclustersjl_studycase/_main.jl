import Pkg
Pkg.instantiate()

using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Dates
using CUDA

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")
include("src/models/get_model.jl")

nprocs = 1
learning_rate = 0.001
batch_size = 32
iterations_per_client = 200

# Define model
# model_name = "custom"
# model_name = "vgg16"
model_name = "resnet18"
# model_name = "mobilenetv3_small"
# model_name = "mobilenetv3_large"

model, img_dims = get_model(model_name)
# Download dataset
train_data = CIFAR10(split=:train)[:]
test_data = CIFAR10(split=:test)[:]


# Define data loaders
test_loader = dataset_loader(test_data,
  batch_size=batch_size,
  img_dims=img_dims,
  #n_batches=10000,
)
partial_test_loader = dataset_loader(test_data,
  batch_size=batch_size,
  img_dims=img_dims,
  #n_batches=iterations_per_client,
) # Run test using less instances

### Inicitalizate client nodes ###
# all nodes but master (id=1) are clients
println("Initializating clients")
addprocs(nprocs)

#for w in workers()
#  train_data_client = fetch_partition(train_data, w, nprocs)
#  @everywhere [i] global train_data = $train_data_client
#end

@everywhere workers() begin
  import Pkg
  Pkg.instantiate()

  using Distributed
  using CUDA
  using Flux
  using Metalhead
  using ProgressLogging

  # CUDA.device!(myid() % 2)   ***
  println("Initializating client $(myid())")
  # Load dependencies
  include("src/train_client.jl")
  include("src/dataset_loader.jl")

  learning_rate = $learning_rate
  batch_size = $batch_size
  img_dims = $img_dims
  iterations_per_client = $iterations_per_client

  function fetch_partition(dataset, i, nprocs)
    @assert i>=1 && i<=nprocs
    size_partition = div(length(dataset[:targets]),nprocs)
    a = (i-1)*size_partition + 1
    b = i*size_partition 
    @info a,b
    return (features = dataset[:features][:,:,:,a:b], targets = dataset[:targets][a:b])
  end
  
  # Create local data loader
  train_data = fetch_partition(CIFAR10(split=:train)[:], myid()-1, $nprocs)
  @info "===> $(length(train_data[:targets]))"

  train_loader = dataset_loader(train_data,
    batch_size=batch_size,
    img_dims=img_dims,
    #n_batches=iterations_per_client,
  )

  train_client(model) = train_client(model, train_loader)
end

println("Log initial test accuracy")
initial_timestamp = now()
@time log_model_accuracy(model |> gpu, partial_test_loader; iteration=0, timestamp=now() - initial_timestamp)

# Begin distributed training
println("Start training")
num_iterations = 100

@profile @showprogress for it in 1:num_iterations
  global model

  # Send global model to clients and start the training
  f_references = map(workers()) do client_id
    remotecall(train_client, client_id, model)
  end
  
  # Collect the result of the client training
  local_models = map(f_references) do client_model_future_ref
    fetch(client_model_future_ref)
  end

  # Aggregate clients' results
  model = aggregate(local_models) 

  # Print partial accuracy
  @time log_model_accuracy(model |> gpu, partial_test_loader; iteration=it, timestamp=now() - initial_timestamp)
end

println("Full test accuracy:")
log_model_accuracy(model |> gpu, test_loader; iteration=num_iterations, timestamp=now() - initial_timestamp)

