import StaticPolynomials
const SP = StaticPolynomials
import HomotopyContinuation
const HC = HomotopyContinuation
import MultivariatePolynomials
const MP = MultivariatePolynomials

"""
    SPHomotopy(polynomials, vars) <: AbstractSystem

Create a system using the [`StaticPolynomials`](https://github.com/JuliaAlgebra/StaticPolynomials.jl) package.
Note that `StaticPolynomials` leverages Julias metaprogramming capabilities to automatically generate
functions to evaluate the system and its Jacobian. These generated functions are *very fast* but
at the cost of possibly large compile times. The compile time depends on the size of the support of the polynomial system.
If you intend to solve a large system or you need to solve a system
with the *same support* but different coefficients even large compile times can be worthwile.
As a general rule of thumb this usually is twice as fast as solving the same system using [`FPSystem`](@ref).

## Example
You can use `SPHomotopy` as follows with solve
```julia
@polyvar x y
F = [x^2+3y^4-2, 2y^2+3x*y+4]
solve(F, system=SPHomotopy)
```
"""
struct SPHomotopy{S<:SP.PolynomialSystem} <: HC.AbstractHomotopy
    system::S
end

function SPHomotopy(polys::Vector{<:MP.AbstractPolynomial}, t::MP.AbstractVariable)
    SPHomotopy(SP.PolynomialSystem(polys; parameters=[t]))
end

Base.size(H::SPHomotopy) = (SP.npolynomials(H.system), SP.nvariables(H.system))
HC.cache(H::SPHomotopy, x, p=nothing) = HC.HomotopyNullCache()
HC.evaluate!(u, H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.evaluate!(u, H.system, x, t)
HC.evaluate(H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.evaluate(H.system, x, t)
HC.jacobian!(U, H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.jacobian!(U, H.system, x, t)
HC.jacobian(H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.jacobian(H.system, x, t)
HC.evaluate_and_jacobian!(u, U, H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.evaluate_and_jacobian!(u, U, H.system, x, t)
HC.evaluate_and_jacobian(H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.evaluate_and_jacobian(H.system, x, t)
HC.dt!(U, H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.differentiate_parameters!(U, H.system, x, t)
HC.dt(H::SPHomotopy, x, t, ::HC.HomotopyNullCache) = SP.differentiate_parameters(H.system, x, t)[:,1]
