using HomotopyContinuation, LinearAlgebra, DynamicPolynomials
using Plots, Makie

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

lambda=[exp(i/360*2*pi*im) for i in 1:360];

function findDeltasAndAngles(xayb)
        @info "calculating complex Coupler curve"
        global lambda, System
        results=[];
        for i in 1:360
                Sys=[subs(f, [x;x̂;a;â;y;ŷ;b;b̂]=>xayb, λ=>lambda[i]) for f in System];
                result=solve(Sys);
                push!(results,solutions(result));
        end
        results
end

#help function: finds real coordinates
function isoToReal(a ::ComplexF64,b ::ComplexF64)
        x1=reim(a);
        x2=reim(b);
        if abs(x1[1]-x2[1])<1e-10
                if abs(x1[2]+x2[2])<1e-10
                        return x1
                end
        end
        return -1
end

#help function: filter real solutions
function realResults(res,lambda=lambda)
        @info "filtering real part of the curve"
        results = [];
        for i in 1:length(res)
                for j in 1:2
                        q = isoToReal(res[i][j][1], res[i][j][2])
                        if q!=-1
                                if abs(abs(res[i][j][3])-1)<1e-10
                                        if abs(abs(res[i][j][4])-1)<1e-10
                                                push!(results,[q;lambda[i];res[i][j][3];res[i][j][4]])
                                        end
                                end
                        end
                end
        end
        return results;
end

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


function plot(realRes, bars, points)
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


function plotFourbar(bars,points)
        deltasAngles=findDeltasAndAngles(bars);
        realRes=realResults(deltasAngles);
        plot(realRes, bars, points)
end

function gifFourbar(bars,points)
        deltasAngles=findDeltasAndAngles(bars);
        realRes=realResults(deltasAngles);
        make_gif(realRes, bars, points)
end


function make_gif(realRes, bars, points)
        @info "creating gif"
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
        @gif for t in 1:round((N/2))-1
                t = convert(Int64,t);
                plot = Plots.scatter(AB_points, color=:blue, markersize=1, xlim=(-10,25), ylim=(-10,25))
                Plots.scatter!(plot,P_points, color = :black, markersize=1);
                Plots.scatter!(plot,NinePoints, markersize=3, color=:green)
                lines = [AB_points[1],D_points[2*t-1],P_points[2*t-1],C_points[2*t-1],AB_points[2],C_points[2*t-1],D_points[2*t-1],AB_points[1],D_points[2*t],P_points[2*t],C_points[2*t],AB_points[2],C_points[2*t],D_points[2*t]];
                linecolors=[:blue for i in 1:14]
                Plots.plot!(plot, lines, color = linecolors, linewidth = 2)
        end
end

function plotPng(realRes, bars, points)
        @info "creating png"
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
        scene = Scene( resolution = (1500,1500));
        #angle points A and B
        Makie.scatter!(scene,AB_points, color=:black, markersize=1, marker='▲')
        #Coupler Curve
        Makie.scatter!(scene,P_points, color = :black);
        #Nine points given
        Makie.scatter!(scene,NinePoints, marker='★', markersize=1, color=:green)

        t=1
        linien = [AB_points[1],D_points[2*t-1],P_points[2*t-1],C_points[2*t-1],AB_points[2],C_points[2*t-1],D_points[2*t-1]];
        scene=Makie.lines!(scene, linien, color = :red, linewidth = 3)
        t=30
        linien = [AB_points[1],D_points[2*t-1],P_points[2*t-1],C_points[2*t-1],AB_points[2],C_points[2*t-1],D_points[2*t-1]];
        scene=Makie.lines!(scene, linien, color = :blue, linewidth = 3)
        save("plot2.jpg", scene)
end

#### tests ####
function testPlot()
        bars=convert(Array{ComplexF64},[-1-2*im, -1+2*im, -5*im, 5*im, 3, 3, 4-5*im, 4+5*im]);
        points=[Point2f0(0,5)];
        plotFourbar(bars,points)
end

function testGif()
        bars=convert(Array{ComplexF64},[-1-2*im, -1+2*im, -1-5*im, -1+5*im, 2-2*im, 2+2*im, 2-4*im, 2+4*im]);
        points=[Point2f0(1,5)];
        gifFourbar(bars,points)
end

function testPng()
        bars=convert(Array{ComplexF64},[-1-2*im, -1+2*im, -1-5*im, -1+5*im, 2-2*im, 2+2*im, 2-4*im, 2+4*im]);
        points=[Point2f0(1,5)];
        deltasAngles=findDeltasAndAngles(bars);
        realRes=realResults(deltasAngles);
        plotPng(realRes, bars, points)
end
