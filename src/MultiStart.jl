module MultiStart
using SpecialFunctions
using LinearAlgebra

export mlsl

struct Sequential end
struct Threading end
struct SearchBox{T1, T2, T3, T4, T5}
    volume::T1
    lower::T2
    upper::T3
    bounds::T4
    widths::T5
end
function SearchBox(lower, upper)
    bounds = [(l, u) for (l, u) in zip(lower, upper)]
    widths = [u-l for (l, u) in zip(lower, upper)]
    volume = *(widths...)
    SearchBox(volume, lower, upper, bounds, widths)
end
Base.eltype(::SearchBox{T, <:Any, <:Any, <:Any, <:}) where T = T
function mlsl(obj, L, lower, upper; N, sigma, try_ratio, maxiter, getmin, getminf, partype=Sequential(), try_center=true)

    sbox = SearchBox(lower, upper)

    T = eltype(lower)
    n = length(lower)

    X, fX, Xfrom, fXfrom, star = initial_arrays(L, obj, sbox, try_center, getmin, getminf)

    for k = 1:maxiter
        rk = rₖ(T, n, k, N, sbox, sigma)
        nbest = try_ratio*k*N
        nbest = ceil(Int, nbest) # since try_ratio<1.0 we can safely round up
        X, fX = reduced_set!(fX, X, partype, sbox, obj, N)
        Xr, fXr = X[1:nbest], fX[1:nbest]
        for (xsearch, fxsearch) in zip(Xr, fXr)
            # Skip if in a cluster
            in_cluster(star.X, star.fX, xsearch, fxsearch, rk) && continue

            lxr = L(obj, lower, upper, xsearch)

            Xfrom, fXfrom, star = update_arrays!(fXfrom, Xfrom, star,
             xsearch, fxsearch, getmin(lxr), getminf(lxr))
        end
    end
    star.X, star.fX, Xfrom, fXfrom
end
function initial_arrays(L, obj, sbox, try_center, getmin, getminf)
    if try_center
        center = [l+(u-l)/2 for (l, u) in zip(sbox.lower, sbox.upper)]
        obj_center = obj(center)
        fXfrom = [obj_center]
        Xfrom = [center]
        X = [center]
        fX = [obj_center]
        lxr = L(obj, sbox.lower, sbox.upper, center)
        Xstar = [getmin(lxr)]
        fXstar = [getminf(lxr)]
    else
        # These and the accompanying methods allow for full inferrability of the
        # arrays and of the code in the main loop with the methods defined below.
        X = nothing
        fX = nothing
        Xfrom = nothing
        fXfrom = nothing
        Xstar = nothing
        fXstar = nothing
    end
    X, fX, Xfrom, fXfrom, (X=Xstar, fX=fXstar)
end
function update_arrays!(fXfrom::Nothing, Xfrom::Nothing, star, xsearch, fxsearch, lxr_minx, lxr_min)
    return [xsearch], [fxsearch], (X=[lxr_minx], fX=[lxr_min])
end
function update_arrays!(fXfrom, Xfrom, star, xsearch, fxsearch, lxr_minx, lxr_min)
    push!(fXfrom, fxsearch)
    push!(Xfrom, xsearch)
    push!(star.X, lxr_minx)
    push!(star.fX, lxr_min)
    return Xfrom, fXfrom, star
end

in_cluster(star::T, xsearch, fxsearch, rk) where T <: NamedTuple{<:Any, Tuple{Nothing, Nothing}} = false
function in_cluster(star, xsearch, fxsearch, rk)
    break_now = false
    for (x, fx) in zip(star.X, star.fX)
        is_close = norm(xsearch-x) <= rk
        Δf = fxsearch - fx
        is_larger = fxsearch >= fx
        if is_close && is_larger
            break_now = true
        end
        if break_now
            return true
        end
    end
    false
end

function reduced_set!(fX::Nothing, X::Nothing, partype, sbox, obj, N)
    # should do this in a sorted way
    X = sample_N(sbox.bounds, N)
    fX = obj.(X)
    p = sortperm(fX)
    X .= X[p]
    fX .= fX[p]
    X, fX
end
function reduced_set!(fX, X, partype, sbox, obj, N)
    # should do this in a sorted way
    Xnew = sample_N(sbox.bounds, N)
    push!(X, Xnew...)
    fXnew = obj.(Xnew)
    push!(fX, fXnew...)
    p = sortperm(fX)
    X .= X[p]
    fX .= fX[p]
    X, fX
end
function sample_N(bounds, N)
    [[rand()*(u-l)+l for (l, u) in bounds] for i = 1:N]
end
function rₖ(T, n, k, N, sbox, sigma)
    V = sbox.volume
    1/√T(π)*(gamma(T(1)+T(n)/2)*V*(sigma*log(k*N)/k*N))^(1/T(n))
end

end # module
