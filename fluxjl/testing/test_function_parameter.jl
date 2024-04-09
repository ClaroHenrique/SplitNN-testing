

function pmap_mock(f::Function, ids)
    results = []
    for id in ids
        println("Master node $(id) is working...")
        push!(results, f(id))
    end

    results
end

mapando(f::Function, x) = f(x)

y = mapando(3) do x
    x * 2
end

println(y)
