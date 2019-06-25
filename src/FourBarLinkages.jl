module FourBarLinkages

export compute_generic_solutions, save_generic_solutions, load_generic_solutions

using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using StaticArrays
import JLD2, FileIO

function equations()
    @polyvar x x̂ a â y ŷ b b̂
    @polyvar γ[1:8] γ̂[1:8] δ[1:8] δ̂[1:8]
    #system of polynomials
    D1 = [(â*x-δ̂[i]*x)*γ[i]+(a*x̂-δ[i]*x̂)*γ̂[i]+(â-x̂)*δ[i]+(a-x)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
    D2 = [(b̂*y-δ̂[i]*y)*γ[i]+(b*ŷ-δ[i]*ŷ)*γ̂[i]+(b̂-ŷ)*δ[i]+(b-y)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
    D3 = [γ[i]*γ̂[i]+γ[i]+γ̂[i] for i in 1:8]
    (F=[D1;D2;D3], xayb=[x, x̂, a, â, y, ŷ, b, b̂], γ=γ, γ̂=γ̂, δ=δ, δ̂=δ̂)
end

function compute_start_pair()
    eqs = equations()
    α = rand(8)
    Γ, Γ̂ = cis.(α.*2π) .- 1, cis.(α.*(-2).*π) .- 1
    xayb₀ = randn(ComplexF64,8)

    start_help = map(1:8) do i
        Dᵢ = [eqs.F[i], eqs.F[i+8]]
        Gᵢ = [subs(f, eqs.xayb => xayb₀, eqs.γ=>Γ,eqs.γ̂=>Γ̂) for f in Dᵢ];
        first(solutions(solve(Gᵢ)))
    end

    δ₀, δ̂₀ = first.(start_help), last.(start_help)
    [xayb₀;Γ;Γ̂], [δ₀; δ̂₀]
end

function compute_γ_γ̂(x, x̂, a, â, y, ŷ, b, b̂, δ, δ̂)
    γ = @MVector zeros(eltype(x), 8)
    γ̂ = @MVector zeros(eltype(x), 8)
    for j in 1:8
        Aⱼ = @SMatrix [(â-δ̂[j])*x (a-δ[j])*x̂; (b̂-δ̂[j])*y (b-δ[j])*ŷ]
        cⱼ = -@SVector [δ[j]*(â-x̂)+δ̂[j]*(a-x)-δ[j]*δ̂[j],
                       δ[j]*(b̂-ŷ)+δ̂[j]*(b-y)-δ[j]*δ̂[j]]
        γⱼ, γ̂ⱼ = Aⱼ \ cⱼ
        γ[j] = γⱼ
        γ̂[j] = γ̂ⱼ
    end
    SVector(γ), SVector(γ̂)
end

r(x,a,y,b) = ((x-a)*y/(x-y), (b*x - a*y)/(x-y), a-x, a)
function robert_cognates(v, δ, δ̂)
    x, x̂, a, â, y, ŷ, b, b̂ = view(v, 1:8)
    x₁, a₁, y₁, b₁ = r(x, a, y, b)
    x̂₁, â₁, ŷ₁, b̂₁ = r(x̂, â, ŷ, b̂)
    γ₁, γ̂₁ = compute_γ_γ̂(x₁, x̂₁, a₁, â₁, y₁, ŷ₁, b₁, b̂₁, δ, δ̂)
    x₂, a₂, y₂, b₂ = r(x₁, a₁, y₁, b₁)
    x̂₂, â₂, ŷ₂, b̂₂ = r(x̂₁, â₁, ŷ₁, b̂₁)
    γ₂, γ̂₂ = compute_γ_γ̂(x₂, x̂₂, a₂, â₂, y₂, ŷ₂, b₂, b̂₂, δ, δ̂)
    v₁ = [x₁, x̂₁, a₁, â₁, y₁, ŷ₁, b₁, b̂₁, γ₁..., γ̂₁...]
    v₂ = [x₂, x̂₂, a₂, â₂, y₂, ŷ₂, b₂, b̂₂, γ₂..., γ̂₂...]
    v₁, v₂
end
symmetry(s) = [s[5],s[6],s[7],s[8],s[1],s[2],s[3],s[4],
               s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],
               s[17],s[18],s[19],s[20],s[21],s[22],s[23],s[24]]

compute_generic_solutions() = compute_generic_solutions(compute_start_pair()...)
function compute_generic_solutions(x₀, p₀)
    eqs = equations()
    δ₀, δ̂₀ = p₀[1:8], p₀[9:16]
    group_actions = GroupActions(symmetry, s -> robert_cognates(s, δ₀, δ̂₀))
    @info "Computing all 1442 generic solutions..."
    result = monodromy_solve(eqs.F, x₀, p₀;
                parameters=[eqs.δ;eqs.δ̂],
                group_actions=group_actions,
                target_solutions_count=1442,
                equivalence_classes=true)
end

function save_generic_solutions(result::MonodromyResult, filename::String)
    δ₀, δ̂₀ = result.parameters[1:8], result.parameters[9:16]
    FileIO.save(endswith(filename, ".jld2") ? filename : filename * ".jld2",
                "δ₀", δ₀, "δ̂₀", δ̂₀,
                "solutions", reduce(hcat, result.solutions))
end

function load_generic_solutions(filename)
    data = FileIO.load(endswith(filename, ".jld2") ? filename : filename * ".jld2")
    (solutions=data["solutions"], δ₀=data["δ₀"], δ̂₀=data["δ̂₀"])
end

end # module
