import ChainRules: rrule

function integrate(f, p)
    return mapreduce(+, zip(x, dx)) do (x, dx)
        f(x, p)*dx
    end
end

# Reverse rule for d/dp ∫f(x; p) dx
function rrule(::typeof(integrate), f, p)
    ∇f(x, p) = gradient(f, x, p)[2]  # Gradient w.r.t. p
    I  = integrate( f, p)
    ∇I = integrate(∇f, p)
    function int_pullback(Ī)
        # Only return differential w.r.t. p, since we integrate over x
        return NoTangent(), NoTangent(), Ī.*∇I
    end
    return I, int_pullback
end

function main(p)
    f(x, p) = p[1]*x + p[2]
    return integrate(f, p)
end

dx = diff(0:0.01:1)
x = cumsum(dx)
p = [2, 1]
gradient(main, p)
