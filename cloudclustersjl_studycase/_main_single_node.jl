using ProgressMeter
using Profile
using CUDA, cuDNN
using Flux, MLDatasets
using Flux: train!, onehotbatch
using Dates
using MLUtils
using Metalhead

include("src/utils.jl")
include("src/aggregate.jl")
include("src/models/custom.jl")
include("src/train_client.jl")
include("src/dataset_loader.jl")
include("src/models/get_model.jl")

function fetch_partition(ds, i, nprocs)
  @assert i>=1 && i<=nprocs
  size_partition = div(length(ds[:targets]),nprocs)
  a = (i-1)*size_partition + 1
  b = i*size_partition 
  @info a,b
  return (features = ds[:features][:,:,:,a:b], targets = ds[:targets][a:b])
end

function train_the_model(model_name, dataset; learning_rate=0.001, batch_size=32, partition = (1,1))

  i,n = partition

  # Get the model
  model, img_dims = get_model(model_name)

  # Download dataset
  train_data = fetch_partition(dataset(split=:train)[:], i, n)
  test_data = dataset(split=:test)[:]

  # Define data loaders
  train_loader = dataset_loader(train_data,
    batch_size=batch_size,
    img_dims=img_dims,
    #n_batches=iterations_per_client,
  )
  test_loader = dataset_loader(test_data,
    batch_size=batch_size,
    img_dims=img_dims,
    #n_batches=10000,
  )

  # Log model initial test accuracy
  initial_timestamp = now()
  log_model_accuracy(model |> gpu, test_loader; iteration=0, timestamp=now() - initial_timestamp)

  # Begin distributed training
  println("Start training")
  num_interations = 100

  training_time = Millisecond(0)
  accuracy_time = Millisecond(0)

  @profile @showprogress for it in 1:num_interations

    interation_training_time = now()

    model = train_client(model, train_loader, learning_rate)

    interation_training_time = now() - interation_training_time

    interation_accuracy_time = now()
    
    log_model_accuracy(model |> gpu, test_loader; iteration=it, timestamp=now()-initial_timestamp)

    interation_accuracy_time = now() - interation_accuracy_time

    training_time += interation_training_time
    accuracy_time += interation_accuracy_time

    @info "iteration training time: $(interation_training_time), overall training time: $(training_time)"
    @info "iteration accuracy time: $(interation_accuracy_time), overall accuracy time: $(accuracy_time)"
  end

end