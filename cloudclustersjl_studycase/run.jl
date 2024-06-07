using Distributed

include("_main.jl")

addprocs(["ubuntu@52.200.154.101", "ubuntu@52.55.231.222", "ubuntu@54.210.151.84", "ubuntu@52.201.143.56"]; sshflags=`-i /home/heron/hpc-shelf-credential.pem`, exename=`/home/ubuntu/.juliaup/bin/julia`, dir=`/home/ubuntu/heron/SplitNN-testing/cloudclustersjl_studycase`, tunnel=true)

train_the_model(:resnet18,CIFAR10)
