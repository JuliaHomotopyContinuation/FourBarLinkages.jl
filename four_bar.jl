using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using StaticArrays
using JLD2

@load "four_bar_start_solution.jld2"


@polyvar x x̂ a â y ŷ b b̂
@polyvar γ[1:8] γ̂[1:8]
@polyvar δ[1:8] δ̂[1:8]

#variable groups
vargroups=[[x,x̂,a,â],[y,ŷ,b,b̂]];
for i in 1:8
    push!(vargroups, [γ[i], γ̂[i]])
end

#system of polynomials
D1 =[(â*x-δ̂[i]*x)*γ[i]+(a*x̂-δ[i]*x̂)*γ̂[i]+(â-x̂)*δ[i]+(a-x)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
D2 =[(b̂*y-δ̂[i]*y)*γ[i]+(b*ŷ-δ[i]*ŷ)*γ̂[i]+(b̂-ŷ)*δ[i]+(b-y)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
D3 =[γ[i]*γ̂[i]+γ[i]+γ̂[i] for i in 1:8]
FSystem=[D1;D2;D3]

#finding one feasible solution
@info "Calculating one start solution"
Γ_help=rand(8);
Γ=[exp(Γ_help[i]*2*pi*im)-1 for i in 1:8]
Γ̂=[exp(-Γ_help[i]*2*pi*im)-1 for i in 1:8]
xayb_rand = randn(ComplexF64,8)
results=[];
for i in 1:8
        F=[D1[i];D2[i]]
        startF=[subs(f,[x;x̂;a;â;y;ŷ;b;b̂]=>xayb_rand,γ=>Γ,γ̂=>Γ̂) for f in F];
        result = solve(startF);
        push!(results, solutions(result));
end

#finding all feasible solutions
@info "Using Monodromy to find all possible start solutions"
start_help=[results[i][1] for i in 1:8];
δ₀ = [start_help[i][1] for i in 1:8]
δ̂₀ = [start_help[i][2] for i in 1:8]
start_par=[δ₀; δ̂₀];
start_var=[xayb_rand;Γ;Γ̂];

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


G = SPSystem(FSystem; parameters=[δ;δ̂])

#GroupActions
#Symmetry
group_actions = let δ₀=δ₀, δ̂₀=δ̂₀
    GroupActions(s->[s[5],s[6],s[7],s[8],s[1],s[2],s[3],s[4],
                s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],
                s[17],s[18],s[19],s[20],s[21],s[22],s[23],s[24]],
                s -> robert_cognates(s, δ₀, δ̂₀))
end
start_sol=monodromy_solve(FSystem,start_var, start_par;
                        parameters=[δ;δ̂],
                        group_actions=group_actions,
                        target_solutions_count=1442,
                        equivalence_classes=true);


using JLD2, FileIO

four_bar_start_solution = Dict(["δ₀" => δ₀,
            "δ̂₀" => δ̂₀,
            "solutions"=> reduce(hcat, start_sol.solutions)])

save("four_bar_start_solution.jld2",
            "δ₀", δ₀,
            "δ̂₀", δ̂₀,
            "solutions", reduce(hcat, start_sol.solutions))
load("four_bar_start_solution.jld2")

start_var=[xayb_rand;Γ;Γ̂];

start_sol=monodromy_solve(FSystem,start_var, start_par;
                        parameters=[δ;δ̂],
                        group_actions=S,
                        target_solutions_count=4326,
                        timeout=600,
                        equivalence_classes=true);
sols_complex_parameters = convert(Array{Vector{ComplexF64}}, solutions(start_sol));


#finding FourBars given 9 points
function fourBar(array)
        @info "Tracking solutions from start solutions"
        global sols_complex_parameters, FSystem, start_par, vargroups
        target_p=[array[i][1]+array[i][2]*im - (array[1][1]+array[1][2]*im) for i in 2:9]
        target_phat=conj(target_p);
        totalresult=solve(FSystem, sols_complex_parameters, parameters=[δ;δ̂], start_parameters=start_par, target_parameters=[target_p;target_phat], variable_groups=vargroups);
        results=solutions(totalresult)
end

#help funtion to turn isotropic coordinates into real coordinates if possible
function isoToReal(a ::ComplexF64,b ::ComplexF64)
         x1=reim(a);
         x2=reim(b);
         if abs(x1[1]-x2[1])<1e-10
                if abs(x1[2]+x2[2])<1e10
                         return x1
                end
        end
        return -1
 end

#filter real solutions
 function realFourbars(res)
         @info "Finding all real solutions"
         results=[]
         for i in 1:length(res)
                 q = [isoToReal(res[i][2*j-1], res[i][2*j]) for j in 1:4]
                 if findall(x->x==-1, q)==[]
                         push!(results,res[i][1:8])
                 end
        end
        return results;
  end


function findRealFourbars(array)
        realFourbars(fourBar(array))
end
