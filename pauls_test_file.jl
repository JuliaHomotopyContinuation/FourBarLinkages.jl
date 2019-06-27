using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using Plots, Makie

points=[Point2f0(0.8961867,-0.09802917),
Point2f0(1.2156535, -1.18749100),
Point2f0(1.5151435, -0.85449808),
Point2f0(1.6754775,  -0.48768058),
Point2f0(1.7138690,-0.30099232),
Point2f0(1.7215236,0.03269953),
Point2f0(1.6642029, 0.33241088),
Point2f0(1.4984171, 0.74435576),
Point2f0(1.3011834,  0.92153806)].*10
reals=findRealFourbars(points)
oneSol=reals[1]


@polyvar x x̂ a â y ŷ b b̂ #four bars
@polyvar γ[1:8] γ̂[1:8] #angles
@polyvar τ τ̂ #coupler point differences
@polyvar λ μ θ #angles

System=[
(x-a)*λ-x*θ-τ+a,
(x̂-â)*θ-x̂*λ-(τ̂-â)*λ*θ,
(y-b)*μ-y*θ-τ+b,
(ŷ-b̂)*θ-ŷ*μ-(τ̂-b̂)*μ*θ
]

bars = oneSol
ϕ=rand()*2*pi
Sys=[subs(f, [x;x̂;a;â;y;ŷ;b;b̂]=>bars) for f in System]

result=solve([subs(f,λ=>cis(ϕ)) for f in Sys])
res = solutions(result)
function is_it_real(r)
        q = isoToReal(r[1], r[2])
                if q!=-1
                        if abs(abs(r[3])-1)<1e-10
                                if abs(abs(r[4])-1)<1e-10
                                        return true
                                end
                        end
                end
        return false
end
function real_coords(r, β)
        q = isoToReal(r[1], r[2])
        [q;cis(β);r[3];r[4]]
end

INDICES = findall(is_it_real.(res))


function trace_points(Sys, start_sol_for_moving_ϕ, ϕ, Δt, N)
        tracker = pathtracker(Sys, parameters=[λ], p₁=[cis(ϕ)], p₀=[cis(ϕ+Δt)])
        back_at_the_start = false
        current_point = start_sol_for_moving_ϕ
        curve = [deepcopy(real_coords(current_point, ϕ))]
        β = ϕ
        plus = true
        for i in 1:N
                if plus
                        set_parameters!(tracker; start_parameters=[cis(β)], target_parameters=[cis(β+Δt)])
                else
                        set_parameters!(tracker; start_parameters=[cis(β)], target_parameters=[cis(β-Δt)])
                end
                track!(tracker, current_point)
                if !is_it_real(solution(tracker))
                        result = solve([subs(f,λ=>cis(β)) for f in Sys])
                        res = solutions(result)
                        I = findmax([norm(current_point - s) for s in res])[2]
                        current_point = res[I]
                        push!(curve, deepcopy(real_coords(current_point, ϕ)))
                        plus = !plus

                else
                        current_point = solution(tracker)
                        push!(curve, deepcopy(real_coords(current_point, ϕ)))
                        if plus
                                β += Δt
                        else
                                β -= Δt
                        end
                end
        end
        curve
end



realRes = trace_points(Sys, res[1], ϕ, 0.025, 3000)



plot(realRes, bars, points)
