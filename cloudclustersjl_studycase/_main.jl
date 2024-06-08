using ProgressMeter
using Distributed
using Profile
using Flux, MLDatasets
using Dates
using CUDA
using Metalhead
using MLUtils

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")
include("src/models/get_model.jl")

# Define model
# model_name = "custom"
# model_name = "vgg16"
model_name = "resnet18"
# model_name = "mobilenetv3_small"
# model_name = "mobilenetv3_large"

function train_the_model(model_name, dataset; learning_rate=0.001, batch_size=32)
  train_the_model(model_name, dataset, workers(); learning_rate = learning_rate, batch_size = batch_size)
end

function train_the_model(model_name, dataset, workers; learning_rate=0.001, batch_size=32)
    
  nprocs = length(workers)

  # Get the model
  model, img_dims = get_model(model_name)

  # Download dataset
  train_data = dataset(split=:train)[:]
  test_data = dataset(split=:test)[:]

  # Define data loaders
  test_loader = dataset_loader(test_data,
    batch_size=batch_size,
    img_dims=img_dims,
  )

  datasetsymb = Symbol(CIFAR10)

  @everywhere workers begin

    @eval using Distributed
    @eval using CUDA
    @eval using Flux
    @eval using Metalhead
    @eval using ProgressLogging
    @eval using MLDatasets

    println("Initializating client $(myid())")
    # Load dependencies
    include("src/train_client.jl")
    include("src/dataset_loader.jl")

    learning_rate = $learning_rate
    batch_size = $batch_size
    img_dims = $img_dims

    function fetch_partition(ds, i, nprocs)
      @assert i>=1 && i<=nprocs
      size_partition = div(length(ds[:targets]),nprocs)
      a = (i-1)*size_partition + 1
      b = i*size_partition 
      @info a,b
      return (features = ds[:features][:,:,:,a:b], targets = ds[:targets][a:b])
    end
    
    # Create local data loader
    dataset = getfield(MLDatasets, Symbol($datasetsymb))
    train_data = fetch_partition(dataset(split=:train)[:], myid()-1, $nprocs)
    @info "===> $(length(train_data[:targets]))"

    train_loader = dataset_loader(train_data,
      batch_size=batch_size,
      img_dims=img_dims,
    )

    CUDA.device!(mod(indexin(myid(), procs(myid()))[1],CUDA.ndevices()))
    @info "process $(myid()) using device $(CUDA.device())"

    train_client(model) = train_client(model, train_loader, learning_rate)
  end

  println("computing initial test accuracy ...")
  
  initial_timestamp = now()
  log_model_accuracy(model |> gpu, test_loader; iteration=0, timestamp=now() - initial_timestamp)

  # Begin distributed training
  println("start training !")
  num_iterations = 100

  model = Ref(model)

  training_time = Millisecond(0)
  accuracy_time = Millisecond(0)

  @profile @showprogress for it in 1:num_iterations    

    interation_training_time = now()

    # Send global model to clients and start the training
    f_references = map(workers) do client_id
      remotecall(train_client, client_id, model[])
    end
    
    # Collect the result of the client training
    local_models = map(f_references) do client_model_future_ref
      fetch(client_model_future_ref)
    end

    # Aggregate clients' results
    model[] = aggregate(local_models) 

    interation_training_time = now() - interation_training_time

    interation_accuracy_time = now()
    # Print partial accuracy
    log_model_accuracy(model[] |> gpu, test_loader; iteration=it, timestamp = now() - initial_timestamp)
    interation_accuracy_time = now() - interation_accuracy_time

    training_time += interation_training_time
    accuracy_time += interation_accuracy_time

    @info "iteration training time: $(interation_training_time), overall training time: $(training_time)"
    @info "iteration accuracy time: $(interation_accuracy_time), overall accuracy time: $(accuracy_time)"

  end

end

