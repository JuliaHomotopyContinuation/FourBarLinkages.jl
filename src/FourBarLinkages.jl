module FourBarLinkages

export compute_generic_solutions, save_generic_solutions, load_generic_solutions,
    four_bars, FourBar, animate, configurations

using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using Interpolations
using StaticArrays
using Makie
import Parameters: @unpack
import JLD2, FileIO

include("sp_homotopy.jl")

struct FourBar
    A::ComplexF64
    B::ComplexF64
    x::ComplexF64
    a::ComplexF64
    u::ComplexF64
    y::ComplexF64
    b::ComplexF64
    v::ComplexF64
    P₀::ComplexF64
end

function FourBar(solution::Vector{ComplexF64}, P₀::ComplexF64)
    x, a, y, b = solution[1], solution[3], solution[5], solution[7]
    u = x - a
    v = y - b
    A = P₀ + a
    B = P₀ + b
    FourBar(A, B, x, a, u, y, b, v, P₀)
end

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

compute_generic_solutions(;kwargs...) = compute_generic_solutions(compute_start_pair()...;kwargs...)

jld2_file(filename) = endswith(filename, ".jld2") ? filename : filename * ".jld2"
function compute_generic_solutions(x₀, p₀; filename=nothing)
    eqs = equations()
    δ₀, δ̂₀ = p₀[1:8], p₀[9:16]
    group_actions = GroupActions(symmetry, s -> robert_cognates(s, δ₀, δ̂₀))
    @info "Computing all 1442 generic solutions..."
    result = monodromy_solve(eqs.F, x₀, p₀;
                parameters=[eqs.δ;eqs.δ̂],
                group_actions=group_actions,
                target_solutions_count=1442,
                equivalence_classes=true)
    δ₀, δ̂₀ = result.parameters[1:8], result.parameters[9:16]
    data = Dict(["δ₀" => δ₀, "δ̂₀" => δ̂₀, "solutions" => reduce(hcat, result.solutions)])
    if filename !== nothing
        FileIO.save(jld2_file(filename), data)
    end
    data
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

function solve_instance(δ::Vector{ComplexF64}, filename::String)
    eqs = equations()
    δ̂ = conj.(δ)
    generic_sols, δ₀, δ̂₀ = load_generic_solutions(filename)
    start_sols = [view(generic_sols, 1:24, i) for i in 1:size(generic_sols,2)]
    res = solve(eqs.F, start_sols;
            accuracy=1e-6,
            precision=PRECISION_ADAPTIVE,
            max_lost_digits=10,
            parameters = [eqs.δ; eqs.δ̂],
            start_parameters=[δ₀;δ̂₀],
            target_parameters=[δ;δ̂])
end

is_conjugated_pair(u, v, tol) = abs(u - conj(v)) < tol
is_physical_solution(s, tol) = all(j -> is_conjugated_pair(s[j], s[j+1], tol), 1:2:8)
function physical_four_bars(solutions; tol=1e-10)
    filter(s -> is_physical_solution(s, tol), solutions)
end

function four_bars(points::Vector{<:Complex}, filename::String; real_tol=1e-10)
    @assert length(points) == 9 "Expected 9 points"
    eqs = equations()
    P₀ = points[1]
    δ = points[2:9] .- P₀
    result = solve_instance(δ, filename)

    fourbars = FourBar[]
    for s in solutions(result; only_nonsingular=true)
        if is_physical_solution(s, real_tol)
            push!(fourbars, FourBar(s, P₀))
        end
    end
    fourbars
end

function is_valid_loop_solution(r)
    is_conjugated_pair(r[3], r[4], 1e-4) || return false
    abs(abs(r[1])-1) < 1e-4 && abs(abs(r[2]) - 1) < 1e-4
end


function loop_equations(F::FourBar)
    @unpack x, a, y, b = F
    x̂, â, ŷ, b̂ = conj.((x,a,y,b))
    vars = @polyvar λ μ θ τ τ̂ # and offsets
    [(x-a)*λ-x*θ-τ+a,
     (x̂-â)*θ-x̂*λ-(τ̂-â)*λ*θ,
     (y-b)*μ-y*θ-τ+b,
     (ŷ-b̂)*θ-ŷ*μ-(τ̂-b̂)*μ*θ], vars
end

function trace_points(F::FourBar, λ₀, x₀; Δt = 1e-2, max_steps=20_000, accuracy=1e-8)
    loop, (λ, _) = loop_equations(F)
    ϕ = ϕ₀ = angle(λ₀)
    angles = [(cis(ϕ₀), x₀[1], x₀[2])]
    μ₀, θ₀ = angle(x₀[1]), angle(x₀[2])
    tracker = coretracker(SPHomotopy(loop, λ), [randn(ComplexF64, 4)]; accuracy=accuracy)
    HC.setup!(tracker, x₀, cis(ϕ), cis(ϕ+Δt))
    x = current_x(tracker)
    y = copy(x)
    for i in 2:max_steps
        retcode = track!(tracker, y, cis(ϕ), cis(ϕ+Δt))
        # todo
        if retcode != HC.CoreTrackerStatus.success
            @warn "PathTracker failed with $retcode"
            break
        end

        if is_valid_loop_solution(x)
            ϕ += Δt
            push!(angles, (cis(ϕ), x[1], x[2]))
            y .= current_x(tracker)
        else
            branch_solutions = solutions(solve([subs(f, λ => cis(ϕ)) for f in loop]))
            y .= branch_solutions[last(findmax(norm.([s - y for s in branch_solutions])))]
            Δt = -Δt
            push!(angles, (cis(ϕ), y[1], y[2]))
        end

        if abs(ϕ - ϕ₀) < 0.01Δt && abs(μ₀ - angle(y[1])) < 1e-2 && abs(θ₀ - angle(y[2])) < 1e-2
            break
        end
    end
    angles
end

function δ_angles_pairs(F::FourBar, δ)
    loop, (_, _, _, τ, τ̂) = loop_equations(F)
    pairs = Tuple{ComplexF64, NTuple{3, ComplexF64}}[]
    for δᵢ in δ
        # do only 1:3 since overdetermined sometimes doesn't work
        # I think the randomization is then "bad".
        # Which seems to happen for these equations quite often.
        sols = solutions(solve([subs(f, τ => δᵢ, τ̂ => conj(δᵢ)) for f in loop][1:3]))
        # filter out non physical solutions
        for (λ, μ, θ) in sols
            if isapprox(abs(λ), 1; atol=1e-6) &&
               isapprox(abs(μ), 1; atol=1e-6) &&
               isapprox(abs(θ), 1; atol=1e-6)
                push!(pairs, (δᵢ, (λ, μ, θ)))
            end
        end
    end
    pairs
end

function missed_coupler_points(δ_angles_pairs, angles)
    filter(δ_angles_pairs) do (δ, (λ, μ, θ))
        α = SVector(λ, μ, θ)
        for s in angles
            if norm(α - SVector(s)) < 1e-2
                return false
            end
        end
        true
    end
end

function configurations(F::FourBar, coupler_points::Vector{ComplexF64}; kwargs...)
    δ = coupler_points[2:9] .- coupler_points[1]
    pairs = δ_angles_pairs(F, δ)
    # by the computation is (λ, μ, θ) = (0,0,0) a valid angle configuration
    angles₁ = trace_points(F, 0.0, [1.0, 1.0, 0.0, 0im]; kwargs...)
    curves = [angles₁]
    pairs₂ = missed_coupler_points(pairs, angles₁)
    if !isempty(pairs₂)
        δ, (λ, μ, θ) = pairs₂[1]
        angles₂ = trace_points(F, λ, [μ, θ, δ, conj(δ)]; kwargs...)
        push!(curves, angles₂)
    end
    curves
end


to_point(z::ComplexF64) = Point2f0(reim(z)...)

function four_bar_positions(F::FourBar, angles)
    A = to_point(F.A)
    B = to_point(F.B)
    map(angles) do ((λ, μ, θ))
        (A=A, B=B, C=to_point(F.A + F.u * λ), D=to_point(F.B + F.v * μ),
            P=to_point(F.A + F.u * λ - F.x * θ))
    end
end

function compute_limits(positions)
    xmin = ymin = Float32(Inf)
    xmax = ymax = -Float32(Inf)
    for pos in positions
        xmin = min(xmin, pos.A[1], pos.B[1], pos.C[1], pos.D[1], pos.P[1])
        xmax = max(xmax, pos.A[1], pos.B[1], pos.C[1], pos.D[1], pos.P[1])
        ymin = min(ymin, pos.A[2], pos.B[2], pos.C[2], pos.D[2], pos.P[2])
        ymax = max(ymax, pos.A[2], pos.B[2], pos.C[2], pos.D[2], pos.P[2])
    end
    w = (xmax - xmin) * 1.1
    h = (ymax - ymin) * 1.1
    FRect(xmin, ymin, w, h)
end

function animate(F::FourBar, coupler_points; Δt=1e-3, kwargs...)
    angles = configurations(F, coupler_points; Δt=Δt)
    animate(F, angles, coupler_points; kwargs...)
end
function animate(F::FourBar, angles::Vector{<:Vector}, coupler_points::Vector{ComplexF64})
    animate(F, angles..., coupler_points)
end
function animate(F::FourBar,
        angles::Vector{NTuple{3,ComplexF64}},
        coupler_points::Vector{ComplexF64};
        show_axis=true,
        fps=24, seconds=5,
        loop = false,
        filename::Union{String,Nothing}=nothing)

    positions = four_bar_positions(F, angles)
    P = map(pos -> pos.P, positions)
    #convert vectors to points
    given_points = to_point.(coupler_points)
    #plot Fourbar
    limits = compute_limits(positions);
    markersize = max(limits.widths...)/50
    scene = Scene(limits=limits, resolution = (1500,1500), scale_plot=false);
    #angle points A and B


    Makie.scatter!(scene, [to_point(F.A), to_point(F.B)], color=:DIMGRAY,
                markersize=markersize, marker='▲', show_axis=show_axis)
    Makie.scatter!(scene, given_points, marker=:x, markersize=markersize,
        color=:INDIANRED, show_axis=show_axis)
    #Coupler Curve
    # Makie.lines!(scene, P, color = :black, show_axis=show_axis);

    #Nine points given
    t_source = Node(1)

    loop_closed = false
    curve_at(t) = loop_closed ? (@view P[1:end]) : view(P, 1:t)
    Makie.lines!(scene, lift(curve_at, t_source), color = :black, show_axis=show_axis);

    fourbar_at = t -> begin
        A, B, C, D, Pᵢ = positions[t]
        [A,C,Pᵢ,D,B,D,C]
    end
    lines!(scene, lift(t->fourbar_at(t), t_source), color = :DARKSLATEBLUE, linewidth = 3, show_axis=show_axis)



    n = length(positions)
    curve_partial_lengths = [0;cumsum([norm(P[i] - P[i-1]) for i in 2:length(P)])]
    curve_length = last(curve_partial_lengths)
    itp = interpolate((curve_partial_lengths,), 1:n, Gridded(Linear()))
    N = seconds * fps
    if filename !== nothing
        record(scene, filename, 1:N; framerate=fps) do k
            push!(t_source, round(Int, itp(k/N * curve_length)))
        end
    else
        display(scene)
        if loop
            k = 1
            while true
                push!(t_source, round(Int, itp(k/N * curve_length)))
                sleep(1/24)
                if k == N
                    k = 1
                    loop_closed = true
                else
                    k += 1
                end
            end
        else
            for k in 1:N
                push!(t_source, round(Int, itp(k/N * curve_length)))
                sleep(1/24)
            end
        end
    end
end



function animate(F::FourBar,
        angles1::Vector{NTuple{3,ComplexF64}},
        angles2::Vector{NTuple{3,ComplexF64}},
        coupler_points::Vector{ComplexF64};
        show_axis=true,
        fps=24, seconds=5,
        loop = false,
        filename::Union{String,Nothing}=nothing)

    positions1 = four_bar_positions(F, angles1)
    positions2 = four_bar_positions(F, angles2)
    limits = compute_limits([positions1; positions2])
    markersize = max(limits.widths...) / 50

    scene = Scene(limits=limits, resolution = (1500,1500), scale_plot=false);

    # Draw mechanism ankers and coupler points
    Makie.scatter!(scene, [to_point(F.A), to_point(F.B)],
                color=:DIMGRAY,
                markersize=markersize, marker='▲', show_axis=show_axis)
    Makie.scatter!(scene, to_point.(coupler_points), marker=:x, markersize=markersize,
                color=:INDIANRED, show_axis=show_axis)

    source1, loop_closed_ref1 = add_mechanism!(scene, positions1)
    source2, loop_closed_ref2 = add_mechanism!(scene, positions2; color=:lightseagreen)


    itp1 = interpolate_curve(positions1)
    itp2 = interpolate_curve(positions2)

    N = seconds * fps
    if filename !== nothing
        record(scene, filename, 1:N; framerate=fps) do k
            push!(source1, round(Int, itp1(k/N)))
            push!(source2, round(Int, itp2(k/N)))
        end
    else
        display(scene)
        if loop
            k = 1
            while true
                push!(source1, round(Int, itp1(k/N)))
                push!(source2, round(Int, itp2(k/N)))
                sleep(1/24)
                if k == N
                    k = 1
                    loop_closed_ref1[] = true
                    loop_closed_ref2[] = true
                else
                    k += 1
                end
            end
        else
            for k in 1:N
                push!(source1, round(Int, itp1(k/N)))
                push!(source2, round(Int, itp2(k/N)))
                sleep(1/24)
            end
        end
    end
end


function interpolate_curve(pos)
    n = length(pos)
    partials = [0.0]
    l = 0.0
    for i in 2:n
        l += norm(pos[i].P - pos[i-1].P)
        push!(partials, l)
    end
    # normalize partials to length 1
    partials ./= partials[end]
    itp = interpolate((partials,), 1:n, Gridded(Linear()))
    itp
end

function add_mechanism!(scene, positions; color=:DARKSLATEBLUE)
    loop_closed = Ref(false)
    source = Node(1)
    P = map(pos -> pos.P, positions)
    curve_at(t) = loop_closed[] ? (@view P[1:end]) : view(P, 1:t)
    Makie.lines!(scene, lift(curve_at, source),
                color=color, linewidth=2, show_axis=false);
    fourbar_at = t -> begin
        A, B, C, D, Pᵢ = positions[t]
        [A,C,Pᵢ,D,B,D,C]
    end
    lines!(scene, lift(fourbar_at, source), color=color,
                linewidth = 3, show_axis=false)
    source, loop_closed
end

end # module
