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

function FourBar(x, a, y, b, P₀::ComplexF64)
    u = x - a
    v = y - b
    A = P₀ + a
    B = P₀ + b
    FourBar(A, B, x, a, u, y, b, v, P₀)
end
function FourBar(solution::Vector{ComplexF64}, P₀::ComplexF64)
    x, a, y, b = solution
    FourBar(x, a, y, b, P₀)
end

Base.show(io::IO, F::FourBar) = print_fieldnames(io, F)

"""
     print_fieldnames(io::IO, obj)

 A better default printing for structs.
 """
 function print_fieldnames(io::IO, obj)
     println(io, typeof(obj), ":")
     for name in fieldnames(typeof(obj))
         if getfield(obj, name) !== nothing
             println(io, " • ", name, " → ", getfield(obj, name))
         end
     end
 end

function equations()
    @polyvar x a y b x̂ â ŷ b̂
    @polyvar γ[1:8] γ̂[1:8] δ[1:8] δ̂[1:8]
    #system of polynomials
    D1 = [(â*x-δ̂[i]*x)*γ[i]+(a*x̂-δ[i]*x̂)*γ̂[i]+(â-x̂)*δ[i]+(a-x)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
    D2 = [(b̂*y-δ̂[i]*y)*γ[i]+(b*ŷ-δ[i]*ŷ)*γ̂[i]+(b̂-ŷ)*δ[i]+(b-y)*δ̂[i]-δ[i]*δ̂[i] for i in 1:8]
    D3 = [γ[i]*γ̂[i]+γ[i]+γ̂[i] for i in 1:8]
    (F=[D1;D2;D3], xayb=[x, a, y, b, x̂, â, ŷ, b̂], γ=γ, γ̂=γ̂, δ=δ, δ̂=δ̂)
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

function compute_γ_γ̂(x, a, y, b, δ, x̂, â, ŷ, b̂, δ̂)
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
    x, a, y, b, x̂, â, ŷ, b̂ = v
    xayb₁ = r(x, a, y, b)
    x̂âŷb̂₁ = r(x̂, â, ŷ, b̂)
    γ₁, γ̂₁ = compute_γ_γ̂(xayb₁..., δ, x̂âŷb̂₁..., δ̂)
    xayb₂ = r(xayb₁...)
    x̂âŷb̂₂ = r(x̂âŷb̂₁...)
    γ₂, γ̂₂ = compute_γ_γ̂(xayb₂..., δ, x̂âŷb̂₂..., δ̂)
    v₁ = SVector(xayb₁..., x̂âŷb̂₁..., γ₁..., γ̂₁...)
    v₂ = SVector(xayb₂..., x̂âŷb̂₂..., γ₂..., γ̂₂...)
    v₁, v₂
end
# switch roles of (x,a) and (y,b)
# We have the variable ordering x a y b x̂ â ŷ b̂
# So we need to switch to y b x a ŷ b̂ x̂ â
symmetry(s) = [[s[3],s[4],s[1],s[2],s[7],s[8],s[5],s[6]]; s[9:24]]

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


function load_generic_solutions(filename=joinpath(@__DIR__, "..", "data", "four_bar_start_solutions.jld2"))
    data = FileIO.load(endswith(filename, ".jld2") ? filename : filename * ".jld2")
    (solutions=data["solutions"], δ₀=data["δ₀"], δ̂₀=data["δ̂₀"])
end

function solve_instance(δ::Vector{ComplexF64}, filename::String)
    eqs = equations()
    δ̂ = conj.(δ)
    generic_sols, δ₀, δ̂₀ = load_generic_solutions(filename)
    start_sols = [view(generic_sols, 1:24, i) for i in 1:size(generic_sols,2)]
    res = solve(eqs.F, start_sols;
            precision=PRECISION_ADAPTIVE,
            # max_lost_digits=10,
            parameters = [eqs.δ; eqs.δ̂],
            start_parameters=[δ₀;δ̂₀],
            target_parameters=[δ;δ̂])
end

is_conjugated_pair(u, v, tol) = abs(u - conj(v)) < tol
is_physical_solution(s, tol) = all(j -> is_conjugated_pair(s[j], s[j+4], tol), 1:4)
function physical_four_bars(solutions; tol=1e-10)
    filter(s -> is_physical_solution(s, tol), solutions)
end

function four_bars(points::Vector{<:Complex}, filename::String; kwargs...)
    @assert length(points) == 9 "Expected 9 points"
    eqs = equations()
    P₀ = points[1]
    δ = points[2:9] .- P₀
    result = solve_instance(δ, filename)
    four_bars(result, points; kwargs...)
end
function four_bars(result, points; real_tol=1e-10)
    P₀ = points[1]
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

function trace_points(F::FourBar, λ₀, x₀; Δt = 1e-3, max_steps=20_000, accuracy=1e-8)
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

        if retcode != HC.CoreTrackerStatus.success
            @warn "PathTracker failed with $retcode"
            break
        end

        if is_valid_loop_solution(x)
            ϕ += Δt
            push!(angles, (cis(ϕ), x[1], x[2]))
            y .= current_x(tracker)
        else
            # jump to different branch
            branch_solutions = solutions(solve([subs(f, λ => cis(ϕ)) for f in loop]))
            y .= branch_solutions[last(findmax(norm.([s - y for s in branch_solutions])))]
            Δt = -Δt
            push!(angles, (cis(ϕ), y[1], y[2]))
        end
        if ϕ > π
            ϕ -= 2π
        elseif ϕ < -π
            ϕ += 2π
        end
        if abs(ϕ - ϕ₀) < 0.5Δt &&
           abs(x₀[1] - y[1]) < 1e-3 &&
           abs(x₀[2] - y[2]) < 1e-3
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
            if norm(α - SVector(s)) < 1e-1
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
    w = (xmax - xmin)
    h = (ymax - ymin)
    FRect(xmin - 0.1w, ymin-0.1h, 1.1w, 1.1h)
end

function fix_scene_limits!(scene, limits)
    O = Point2f0(limits.origin...)
    w, h = limits.widths
    O1 = O + Point2f0(0,w)
    O2 = O1 + Point2f0(h,0)
    O3 = O + Point2f0(h,0)
    Makie.lines!(scene, [O,O1,O2,O3, O], color = :transparent, show_axis=false);
end

function static!(scene, F::FourBar, coupler_points; markersize=1.0)
    Makie.scatter!(scene, [to_point(F.A), to_point(F.B)], color=:BLACK,
                markersize=markersize, marker='▲', show_axis=false)
    if coupler_points !== nothing
        Makie.scatter!(scene, to_point.(coupler_points), marker=:x, markersize=markersize,
            color=:INDIANRED, show_axis=false)
    end
    scene
end


function animate(F::FourBar, coupler_points; Δt=1e-3, kwargs...)
    angles = configurations(F, coupler_points; Δt=Δt)
    animate(F, angles, coupler_points; kwargs...)
end
function animate(F::FourBar, angles::Vector{<:Vector}, coupler_points::Vector{ComplexF64}; kwargs...)
    animate(F, angles..., coupler_points; kwargs...)
end
function animate(F::FourBar,
        angles::Vector{NTuple{3,ComplexF64}},
        coupler_points::Union{Nothing,Vector{ComplexF64}}=nothing;
        show_axis=true,
        fps=24, seconds=5,
        color=:CADETBLUE,
        color2=color,
        loop::Union{Int,Bool} = false,
        filename::Union{String,Nothing}=nothing)

    positions = four_bar_positions(F, angles)
    P = map(pos -> pos.P, positions)
    #convert vectors to points
    #plot Fourbar
    limits = compute_limits(positions);
    markersize = max(limits.widths...)/50
    scene = Scene(limits=limits, resolution = (1500,1500), scale_plot=false);
    fix_scene_limits!(scene, limits)

    static!(scene, F, coupler_points; markersize=markersize)


    source, loop_closed_ref = add_mechanism!(scene, positions;
            color=color,
            show_axis=show_axis, markersize=markersize)
    itp = interpolate_curve(positions)
    N = seconds * fps
    if filename !== nothing
        record(scene, filename, 0:N; framerate=fps) do k
            push!(source, round(Int, itp(k/N)))
        end
    else
        display(scene)

        if loop === false
            nloops = 1
        elseif loop === true
            nloops = typemax(Int64)
        else
            nloops = loop::Int
        end

        for _ in 1:nloops
            for k in 0:N
                push!(source, round(Int, itp(k/N)))
                sleep(1/24)
            end
            loop_closed_ref[] = true
        end
    end
end



function animate(F::FourBar,
        angles1::Vector{NTuple{3,ComplexF64}},
        angles2::Vector{NTuple{3,ComplexF64}},
        coupler_points::Union{Nothing,Vector{ComplexF64}}=nothing;
        show_axis=true,
        color=:CADETBLUE,
        color2=:DODGERBLUE,
        fps=24, seconds=5,
        loop::Union{Int,Bool} = false,
        filename::Union{String,Nothing}=nothing)

    positions1 = four_bar_positions(F, angles1)
    positions2 = four_bar_positions(F, angles2)
    limits = compute_limits([positions1; positions2])
    markersize = max(limits.widths...) / 50

    scene = Scene(limits=limits, resolution = (1500,1500), scale_plot=false);

    # Draw mechanism ankers and coupler points
    static!(scene, F, coupler_points; markersize=markersize)

    source1, loop_closed_ref1 = add_mechanism!(scene, positions1; show_axis=show_axis,
                color=color, markersize=markersize)
    source2, loop_closed_ref2 = add_mechanism!(scene, positions2;
                color=color2, show_axis=show_axis, markersize=markersize)


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
        if loop === false
            nloops = 1
        elseif loop === true
            nloops = typemax(Int64)
        else
            nloops = loop::Int
        end

        for _ in 1:nloops
            for k in 0:N
                push!(source1, round(Int, itp1(k/N)))
                push!(source2, round(Int, itp2(k/N)))
                sleep(1/fps)
            end
            loop_closed_ref1[] = true
            loop_closed_ref2[] = true
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

function add_mechanism!(scene, positions; color=:CADETBLUE, markersize=1.0, show_axis=show_axis)
    loop_closed = Ref(false)
    source = Node(1)
    P = map(pos -> pos.P, positions)
    curve_at(t) = loop_closed[] ? (@view P[1:end]) : view(P, 1:t)
    if show_axis == false
        Makie.lines!(scene, P, color = :transparent, show_axis=show_axis);
    end
    Makie.lines!(scene, lift(curve_at, source),
                color=color, linewidth=5, show_axis=show_axis);
    fourbar_at = t -> begin
        A, B, C, D, Pᵢ = positions[t]
        [A,C,Pᵢ,D,B,D,C]
    end
    lines!(scene, lift(fourbar_at, source), color=:black,
                linewidth = 3, show_axis=show_axis)
    scatter!(scene, lift(i -> [P[i]], source), color=color,
        markersize=0.5markersize,
        show_axis=show_axis)
    scatter!(scene, lift(i -> [positions[i].D, positions[i].C], source), color=:black,
        markersize=0.25markersize,
        show_axis=show_axis)
    source, loop_closed
end

end # module
