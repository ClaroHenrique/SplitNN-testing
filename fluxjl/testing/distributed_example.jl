using Distributed

# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(@__DIR__)
  Pkg.instantiate(); Pkg.precompile()
end

# load dependencies in a *separate* @everywhere block
@everywhere begin
  # load dependencies
  using ProgressMeter

  # helper functions
  function process(infile, outfile)
    # read file from disk
    println(infile)
    println(myid())

    # perform calculations
    sleep(3)

    # save new file to disk
    CSV.write(outfile, csv)
  end
end

# MAIN SCRIPT
# -----------

# relevant directories
indir  = joinpath(@__DIR__,"data")
outdir = joinpath(@__DIR__,"results")

# files to process
infiles  = readdir(indir, join=true)
outfiles = joinpath.(outdir, basename.(infiles))
nfiles   = length(infiles)

status = @showprogress pmap(1:nfiles) do i
  try
    process(infiles[i], outfiles[i])
    true # success
  catch e
    false # failure
  end
end

print(status)