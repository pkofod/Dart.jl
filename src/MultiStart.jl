module MultiStart
using SpecialFunctions
using LinearAlgebra

export mlsl

struct Sequential end
struct Threading end
function mlsl(obj, L, lower, upper; N, sigma, gamma, maxiter, getmin, getminf, partype=Sequential(), try_center=true)
    ratio_kN = gamma

    bounds = [(l, u) for (l, u) in zip(lower, upper)]
    widths = [u-l for (l, u) in zip(lower, upper)]
    VD = *(widths...)

    T = eltype(lower)
    n = length(lower)

    if try_center
        center = [l+w/2 for ((l, u), w) in zip(bounds, widths)]
        obj_center = obj(center)
        fXfrom = [obj_center]
        Xfrom = [center]
        X = [center]
        fX = [obj_center]
        lxr = L(obj, lower, upper, center)
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

    for k = 1:maxiter
        rk = rₖ(T, n, k, N, VD, sigma)
        nbest = ratio_kN*k*N
        nbest = ceil(Int, nbest) # since ratio_kN<1.0 we can safely round up
        X, fX = reduced_set!(fX, X, partype, bounds, widths, obj, N)
        Xr, fXr = X[1:nbest], fX[1:nbest]
        for (xsearch, fxsearch) in zip(Xr, fXr)
            # Skip if in a cluster
            in_cluster(Xstar, fXstar, xsearch, fxsearch, rk) && continue

            lxr = L(obj, lower, upper, xsearch)

            Xfrom, fXfrom, Xstar, fXstar = update_arrays!(fXfrom, Xfrom, Xstar, fXstar,
             xsearch, fxsearch, getmin(lxr), getminf(lxr))
        end
    end
    Xstar, fXstar, Xfrom, fXfrom
end

function update_arrays!(fXfrom::Nothing, Xfrom::Nothing, Xstar::Nothing, fXstar::Nothing, xsearch, fxsearch, lxr_minx, lxr_min)
    return [xsearch], [fxsearch], [lxr_minx], [lxr_min]
end
function update_arrays!(fXfrom, Xfrom, Xstar, fXstar, xsearch, fxsearch, lxr_minx, lxr_min)
    push!(fXfrom, fxsearch)
    push!(Xfrom, xsearch)
    push!(Xstar, lxr_minx)
    push!(fXstar, lxr_min)
    return Xfrom, fXfrom, Xstar, fXstar
end

in_cluster(Xstar::Nothing, fXstar::Nothing, xsearch, fxsearch, rk) = false
function in_cluster(Xstar, fXstar, xsearch, fxsearch, rk)
    break_now = false
    for (x, fx) in zip(Xstar, fXstar)
        is_close = norm(xsearch-x) < rk
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

function reduced_set!(fX::Nothing, X::Nothing, partype, bounds, widths, obj, N)
    # should do this in a sorted way
    X = sample_N(bounds, N, widths)
    fX = obj.(X)
    p = sortperm(fX)
    X .= X[p]
    fX .= fX[p]
    X, fX
end
function reduced_set!(fX, X, partype, bounds, widths, obj, N)
    # should do this in a sorted way
    Xnew = sample_N(bounds, N, widths)
    push!(X, Xnew...)
    fXnew = obj.(Xnew)
    push!(fX, fXnew...)
    p = sortperm(fX)
    X .= X[p]
    fX .= fX[p]
    X, fX
end
function sample_N(bounds, N, widths)
    [[rand()*width+b[1] for (b, width) in zip(bounds, widths)] for i = 1:N]
end
function rₖ(T, n, k, N, VD, sigma)
    1/√T(π)*(gamma(T(1)+T(n)/2)*VD*(sigma*log(k*N)/k*N))^(1/T(n))
end

end # module
