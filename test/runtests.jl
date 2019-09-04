using MultiStart, Optim
using Test
function Hosaki(x)
    T = eltype(x)
    pol = Base.Math.@horner(x[1], T(1), -T(8), T(7), -T(7)/3, T(1)/4)

    fx = exp(-x[2])*pol*x[2]^2
end

L = (f, lb, ub, x) -> optimize(f, lb, ub, x, Fminbox(BFGS()))

hosaki_res = mlsl(Hosaki, L, [0.0, 0.0], [10.0, 10.0]; N=5, sigma=4.0, try_ratio=0.1, maxiter=100, getmin=Optim.minimizer, getminf=Optim.minimum)


function HolderTable(x)
    fact1 = sin(x[1])*cos(x[2])
    fact2 = exp(abs(1-sqrt(x[1]^2+x[2]^2)/π))
    -abs(fact1*fact2)
end
holder_res=mlsl(HolderTable, L, [-10.0, -10.0], [10.0, 10.0]; N=5, sigma=4.0, try_ratio=0.1, maxiter=100, getmin=Optim.minimizer, getminf=Optim.minimum)

function Rastrigin(x)
    T = eltype(x)
    d = length(x)
    10*d + sum(xᵢ^2-10*cos(2π*xᵢ) for xᵢ in x)::T
end
n_rastrigin = 3
rastrigin_x, rastrigin_fx, from_x, from_fx=mlsl(Rastrigin, L, fill(-5.12, n_rastrigin), fill(5.12, n_rastrigin); N=5, sigma=4.0, try_ratio=0.1, maxiter=100, getmin=Optim.minimizer, getminf=Optim.minimum)

using LinearAlgebra
function Dropwave(x)
    normx = norm(x)
    num = 1 + cos(12*norm(x))
    den = normx^2/2 + 2
    -num/den
end
dropwave_x, dropwave_fx, from_x, from_fx=mlsl(Dropwave, L, fill(-5.12, 2), fill(5.12, 2); N=5, sigma=4.0, try_ratio=0.1, maxiter=100, getmin=Optim.minimizer, getminf=Optim.minimum, try_center=false)


function Damavandi(x)
    (1-abs(sin(π*(x[1]-2))*sin(π*(x[2]-2))/(π^2*(x[1]-2)*(x[2]-2)))^5)*(2+(x[1]-7)^2+2*(x[2]-7)^2)
end
Damavandi_x, Damavandi_fx, from_x, from_fx=mlsl(Damavandi, L, fill(0.0, 2), fill(14.0, 2); N=5, sigma=4.0, try_ratio=0.1, maxiter=100, getmin=Optim.minimizer, getminf=Optim.minimum, try_center=false)

@testset "MultiStart.jl" begin

end
