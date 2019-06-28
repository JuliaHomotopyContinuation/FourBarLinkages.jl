import Pkg; Pkg.activate(@__DIR__);
using HomotopyContinuation
using FourBarLinkages

using RigidBodyDynamics, StaticArrays, LinearAlgebra
using MeshCatMechanisms

coupler_points = [complex(0.8961867,-0.09802917),
        complex(1.2156535, -1.18749100),
        complex(1.5151435, -0.85449808),
        complex(1.6754775,  -0.48768058),
        complex(1.7138690,-0.30099232),
        complex(1.7215236,0.03269953),
        complex(1.6642029, 0.33241088),
        complex(1.4984171, 0.74435576),
        complex(1.3011834,  0.92153806)]
fourbars = four_bars(coupler_points, "four_bar_start_solution.jld2")

fourbar = fourbars[2]

xayb(F::FourBar) = [F.x, conj(F.x), F.a, conj(F.a), F.y, conj(F.y), F.b, conj(F.b)]

@polyvar x x̂ a â y ŷ b b̂ #four bars
@polyvar γ[1:8] γ̂[1:8] #angles
@polyvar τ τ̂ #coupler point differences
@polyvar λ μ θ #angles

δ₁ = coupler_points[2] - coupler_points[1]
x, x̂, a, â, y, ŷ, b, b̂ = let F=fourbar
    F.x, conj(F.x), F.a, conj(F.a), F.y, conj(F.y), F.b, conj(F.b)
end
τ = δ ₁
τ̂ = conj(δ₁)

loop_equations = [
    (ŷ-b̂)*θ-ŷ*μ-(τ̂-b̂)*μ*θ,
    (x̂-â)*θ-x̂*λ-(τ̂-â)*λ*θ,
    (y-b)*μ-y*θ-τ+b,
    (x-a)*λ-x*θ-τ+a]
θ = ((x-a)*λ-τ+a)/x
loop_equations2 = [
    (ŷ-b̂)*θ-ŷ*μ-(τ̂-b̂)*μ*θ,
    (x̂-â)*θ-x̂*λ-(τ̂-â)*λ*θ,
    (y-b)*μ-τ+b-y*θ]

δ₁ = coupler_points[2] - coupler_points[1]
F = [subs(f, [x;x̂;a;â;y;ŷ;b;b̂]=>xayb(fourbar), τ => δ₁, τ̂ => conj(δ₁)) for f in loop_equations]
F
λ₀, μ₀, θ₀ = first(solutions(solve(F)))

function create_four_bar(sol::FourBar;
    # λ=error("You need to provide λ"),
    # μ=error("You need to provide μ"),
    # θ=error("You need to provide θ"),
    # link masses
    m_1 = 1.0, m_2 = 1.0, m_3 = 1.0,
    # moments of inertia about the center of mass of each link
    I_1 = 0.5, I_2 = 0.5, I_3 = 0.5)

    # gravitational acceleration
    # g = -9.81;
    g = 0.0;

    # link lengths
    l_0 = abs(sol.B - sol.A)
    l_1 = abs(sol.u)
    l_2 = abs(sol.B + sol.v - (sol.A + sol.u))
    l_3 = abs(sol.v)
    @show l_0 l_1 l_2 l_3

    # link center of mass offsets from the preceding joint axes
    c_1 = l_1 / 2
    c_2 = l_2 / 2
    c_3 = l_3 / 2


    # Rotation axis: negative y-axis
    axis = SVector(0., -1., 0.);

    world = RigidBody{Float64}("world")
    fourbar = Mechanism(world; gravity = SVector(0., 0., g))
    joint1 = Joint("joint1", Revolute(axis))
    inertia1 = SpatialInertia(frame_after(joint1),
        com=SVector(c_1, 0, 0),
        moment_about_com=I_1*axis*transpose(axis),
        mass=m_1)
    link1 = RigidBody(inertia1)
    before_joint1_to_world =  Transform3D(
        frame_before(joint1), default_frame(world), SVector(real(sol.A), 0., imag(sol.A)))

    attach!(fourbar, world, link1, joint1,
        joint_pose = before_joint1_to_world)
    joint2 = Joint("joint2", Revolute(axis))
    inertia2 = SpatialInertia(frame_after(joint2),
        com=SVector(c_2, 0, 0),
        moment_about_com=I_2*axis*transpose(axis),
        mass=m_2)
    link2 = RigidBody(inertia2)
    before_joint2_to_after_joint1 = Transform3D(
        frame_before(joint2), frame_after(joint1), SVector(l_1, 0., 0.))
    attach!(fourbar, link1, link2, joint2,
        joint_pose = before_joint2_to_after_joint1)

    joint3 = Joint("joint3", Revolute(axis))
    inertia3 = SpatialInertia(frame_after(joint3),
        com=SVector(l_0, 0., 0.),
        moment_about_com=I_3*axis*transpose(axis),
        mass=m_3)
    link3 = RigidBody(inertia3)
    before_joint3_to_world = Transform3D(
        frame_before(joint3), default_frame(world), SVector(real(sol.B), 0., imag(sol.B)))
    attach!(fourbar, world, link3, joint3, joint_pose = before_joint3_to_world)

    # joint between link2 and link3
    joint4 = Joint("joint4", Revolute(axis))
    before_joint4_to_joint2 = Transform3D(
        frame_before(joint4), frame_after(joint2), SVector(l_2, 0., 0.))
    joint3_to_after_joint4 = Transform3D(
        frame_after(joint3), frame_after(joint4), SVector(-l_3, 0., 0.))
    attach!(fourbar, link2, link3, joint4,
        joint_pose = before_joint4_to_joint2, successor_pose = joint3_to_after_joint4)


    state = MechanismState(fourbar)
    set_configuration!(state, joint1, angle(sol.u)) # λ
    # set_configuration!(state, joint2, angle(sol.B+sol.v - sol.A - sol.u)) # θ
    set_configuration!(state, joint3, angle(sol.v)) #
    set_velocity!(state, joint1, 1)
    set_velocity!(state, joint2, 1)
    set_velocity!(state, joint3, 1)
    # Invalidate the cache variables
    setdirty!(state)
    fourbar, state
end






mech, state = create_four_bar(fourbar)
mvis = MechanismVisualizer(mech, Skeleton(inertias=false))
open(mvis)
ts, qs, vs = simulate(state, 24*20, Δt = 1/24);
qs


ts
qs
vs
MeshCatMechanisms.animate(mvis, ts, qs; realtimerate = 1.)


state = MechanismState(fourbar)
result = DynamicsResult(fourbar)

set_configuration!(state, joint1, 1.6707963267948966) # θ
set_configuration!(state, joint2, -1.4591054166649482) # γ
set_configuration!(state, joint3, 1.5397303602625536) # ϕ




fourbar

using MeshCatMechanisms

mvis = MechanismVisualizer(fourbar, Skeleton(inertias=false))
open(mvis)
setdirty!(state)

ts, qs, vs = simulate(state, 3, Δt = 1e-2);
qs
MeshCatMechanisms.animate(mvis, ts, qs; realtimerate = 1.)


function draw_four_bar!(scene, F::FourBar)
    A, B, C, D = F.A, F.B, F.A+F.u, F.B+F.v
    P = C - F.x
    scatter!(scene, [to_point(A), to_point(B)], marker='▲', markersize=0.5)
    linesegments!(scene,
        [to_point(A)=>to_point(C),
         to_point(C)=>to_point(P),
         to_point(B)=>to_point(D),
         to_point(D)=>to_point(P),
         to_point(C)=>to_point(D)])
    scatter!(scene, [to_point(P)], markersize=0.25)
end

draw_four_bar!(scene, F::FourBar, angles) = draw_four_bar!(scene, F::FourBar, angles...)
function draw_four_bar!(scene, F::FourBar, λ, θ, μ)
    A, B, C, D = F.A, F.B, F.A+abs(F.u)*cis(λ), F.B+abs(F.v)*cis(μ)
    # P = C + abs(F.x)*cis(θ)
    @show abs(A-C)
    @show abs(B-D)
    @show abs(C-D)
    scatter!(scene, [to_point(A), to_point(B)], marker='▲', markersize=0.5)
    linesegments!(scene,
        [to_point(A)=>to_point(C),
         # to_point(C)=>to_point(P),
         to_point(B)=>to_point(D),
         # to_point(D)=>to_point(P),
         to_point(C)=>to_point(D)])
    # scatter!(scene, [to_point(P)], markersize=0.25)
end


four_bar_points(F::FourBar, angles) = four_bar_points(F, angles...)
function four_bar_points(F::FourBar, λ, θ, μ)
    A, B, C, D = F.A, F.B, F.A+abs(F.u)*cis(λ), F.B+abs(F.v)*cis(μ)
    P = C + abs(F.x)*cis(θ)
    A, B, C, D, P
end

four_bar_points(fourbar, qs[1])

using Makie

fourbar
to_point(z::ComplexF64) = Point2f0(reim(z)...)

mech, state = create_four_bar(fourbar)
mvis = MechanismVisualizer(mech, Skeleton(inertias=false))
open(mvis)
setdirty!(state)
ts, qs, vs = simulate(state, 5, Δt = 1e-2);
MeshCatMechanisms.animate(mvis, ts[1:10], qs[1:10]; realtimerate = 0.5)
#plot Fourbar
limits = FRect(-10,-10,20,20);
scene = Scene(resolution=(1500,1500), limits=limits);
draw_four_bar!(scene, fourbar, qs[10])

F = fourbar

λ, θ, μ = qs[1]
set_configuration!(state, qs[1])
setdirty!(state)

RigidBodyDynamics.vertices(mech.tree)

A, B, C, D = F.A, F.B, F.A+abs(F.u)*cis(λ), F.B+abs(F.v)*cis(μ)



P = C + abs(F.x)*cis(-θ)
abs(D-C)

abs(fourbar.u) * cis(λ)

for i in 1:20
    four_bar_points(fourbar, qs[i])
end
