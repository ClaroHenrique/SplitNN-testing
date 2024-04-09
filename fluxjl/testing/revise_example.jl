# ] dev Example

using Revise
using Example
using Distributed

Revise.track(Base)
edit(addprocs)
addprocs
Distributed.amogus

hello("world")

#Example.f("dacueba")

edit(hello)
