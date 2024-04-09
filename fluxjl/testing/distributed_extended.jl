using Distributed

# mv src src2
# ln -s /home/henrique/Documents/teste/Distributed.jl/src src
# cd /Documents/teste/julia/usr/share/julia/compiled/v1.12/Distributed
# rm *


# cd /home/henrique/Documents/teste/SplitNN-testing/fluxjl
# /home/henrique/Documents/teste/julia/julia -p 4 distributed_extended.jl

addprocs(4)
println("1")
@everywhere [4] using MPIClusterManagers
println("2")
fetch(@spawnat 4 addprocs(MPIWorkerManager(2)))
println("3")
@everywhere [4] @everywhere workers() using MPI
println("4")
fetch(@spawnat 4 @everywhere workers() MPI.Init())
println("5")
fetch(@spawnat 4 @everywhere workers() @info MPI.Comm_rank(MPI.COMM_WORLD))
println("6")
fetch(@spawnat 4 @everywhere workers() X=1)
println("7")
fetch(@spawnat 4 @everywhere workers() @info MPI.Reduce(X, (x,y) -> x + y,0,MPI.COMM_WORLD))
println("8")
fetch(@spawnat 4 @everywhere workers() MPI.Finalize())



