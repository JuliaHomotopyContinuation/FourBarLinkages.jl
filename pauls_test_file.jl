using Pkg; Pkg.activate(@__DIR__)

using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using Makie
using FourBarLinkages
xayb(F::FourBar) = [F.x, conj(F.x), F.a, conj(F.a), F.y, conj(F.y), F.b, conj(F.b)]
#
# points=[Point2f0(0.8961867,-0.09802917),
#         Point2f0(1.2156535, -1.18749100),
#         Point2f0(1.5151435, -0.85449808),
#         Point2f0(1.6754775,  -0.48768058),
#         Point2f0(1.7138690,-0.30099232),
#         Point2f0(1.7215236,0.03269953),
#         Point2f0(1.6642029, 0.33241088),
#         Point2f0(1.4984171, 0.74435576),
#         Point2f0(1.3011834,  0.92153806)]
coupler_points = [complex(0.8961867,-0.09802917),
        complex(1.2156535, -1.18749100),
        complex(1.5151435, -0.85449808),
        complex(1.6754775,  -0.48768058),
        complex(1.7138690,-0.30099232),
        complex(1.7215236,0.03269953),
        complex(1.6642029, 0.33241088),
        complex(1.4984171, 0.74435576),
        complex(1.3011834,  0.92153806)] .* 10
reals = four_bars(coupler_points, "four_bar_start_solution.jld2")
bars = xayb(reals[1])


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

ϕ=rand()*2*pi
Sys=[subs(f, [x;x̂;a;â;y;ŷ;b;b̂]=>bars) for f in System]

result=solve([subs(f,λ=>cis(ϕ)) for f in Sys])
res = solutions(result)
function is_it_real(r)
    FourBarLinkages.is_conjugated_pair(r[1], r[2], 1e-10) || return false
    abs(abs(r[3])-1)<1e-10 && abs(abs(r[3])-1)<1e-10
end
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
                        push!(curve, deepcopy(real_coords(current_point, β)))
                        plus = !plus

                else
                        current_point = solution(tracker)
                        push!(curve, deepcopy(real_coords(current_point, β)))
                        if plus
                                β += Δt
                        else
                                β -= Δt
                        end
                end
        end
        curve
end



realRes = trace_points(Sys, res[2], ϕ, 0.025, 3000)

function rotate(x, λ)
        reim((x[1]+x[2]*im)* λ)
end
function toPoints(a)
        P=[]
        for i in 1:length(a)
                push!(P,Point2f0(a[i][1],a[i][2]))
        end
        convert(Array{Point2f0},P)
end

function animate(realRes, bars, points)
        @info "plotting fourbar"
        #realRes = [[(τ[1],τ[2]),λ,μ,θ] for all real results]
        #bars=[x,x̂,a,â,y,ŷ,b,b̂]
        #points=[p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]

        N=length(realRes);

        #initial vectors
        P0=(points[1][1],points[1][2])
        x0=reim(bars[1])
        a0=reim(bars[3])
        y0=reim(bars[5])
        b0=reim(bars[7])
        u0=x0.-a0
        v0=y0.-b0
        A=P0.+a0
        B=P0.+b0

        #calculate position of fourbar at time i
        P=[]
        D=[]
        C=[]
        for i in 1:N
                P_help = P0 .+realRes[i][1]
                push!(P,P_help)
                push!(D,rotate(u0,realRes[i][2]).+A)
                push!(C,rotate(v0,realRes[i][3]).+B)
        end

        #convert vectors to points
        NinePoints=toPoints(points);
        P_points=toPoints(P);
        D_points=toPoints(D);
        C_points=toPoints(C);
        AB_points=[Point2f0(A[1],A[2]), Point2f0(B[1],B[2])];

        #plot Fourbar
        limits = FRect(-5,-5,20,20);
        scene = Scene(limits=limits, resolution = (1500,1500));
        #angle points A and B
        Makie.scatter!(scene,AB_points, color=:black, markersize=1, marker='▲')
        #Coupler Curve
        Makie.scatter!(scene,P_points, color = :black);
        #Nine points given
        Makie.scatter!(scene,NinePoints, marker='★', markersize=1, color=:green)

        #change by time
        # time=Node(1)
        # myfunc(t) = [AB_points[1],D_points[2*t-1],P_points[2*t-1],C_points[2*t-1],AB_points[2],C_points[2*t-1],D_points[2*t-1],AB_points[1],D_points[2*t],P_points[2*t],C_points[2*t],AB_points[2],C_points[2*t],D_points[2*t]];
        # linecolors=[:blue for i in 1:14]
        # scene=Makie.lines!(scene, lift(t->myfunc(t), time), color = linecolors, linewidth = 3)
        #linecolors=[:blue for i in 1:7]
        time=Node(1)
        myfunc(t) = [AB_points[1],D_points[t],P_points[t],C_points[t],AB_points[2],C_points[t],D_points[t]];
        linecolors=[:blue for i in 1:7]
        scene=Makie.lines!(scene, lift(t->myfunc(t), time), color = linecolors, linewidth = 3)
        display(scene)
        for i in 1:div(N,2)
              push!(time,i)
              sleep(1/24)
        end
        #
        # record(scene, "./docs/media/time_series.gif", 1:(N/2)) do i
        #         push!(time,i)
        # end
end

animate(realRes, bars, Point2f0.(reim.(coupler_points)))

realRes
