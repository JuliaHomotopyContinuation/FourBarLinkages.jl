include("four_bar.jl")
include("vis.jl")


function randomexample()
    P=rand(Point2f0,9).*10;
    println(P);
    reals=findRealFourbars(P)
    l=length(reals)
    println(l," real solutions found")
    r=rand(1:l);
    println("Plot solution no. ", r)
    oneSol=reals[r];
    plotFourbar(oneSol,P);
end

function example1()
    # 8 out of 9 points lie near a circle
    P=[Point2f0(0.8961867,-0.09802917),
    Point2f0(1.2156535, -1.18749100),
    Point2f0(1.5151435, -0.85449808),
    Point2f0(1.6754775,  -0.48768058),
    Point2f0(1.7138690,-0.30099232),
    Point2f0(1.7215236,0.03269953),
    Point2f0(1.6642029, 0.33241088),
    Point2f0(1.4984171, 0.74435576),
    Point2f0(1.3011834,  0.92153806)].*10
    println(P);
    reals=findRealFourbars(P)
    l=length(reals)
    println(l," real solutions found")
    r=rand(1:l);
    println("Plot solution no. ", r)
    oneSol=reals[r];
    plotFourbar(oneSol,P);
end

function example2()
    #points in an oval shape
    P=[Point2f0(0.0,0.0), Point2f0(0.30,-0.1), Point2f0(0.70,0.0), Point2f0(0.82,0.2), Point2f0(0.90,0.4), Point2f0(0.90,0.7), Point2f0(0.60,0.7), Point2f0(0.10,0.5), Point2f0(0.00,0.3)].*10;
    println(P);
    reals=findRealFourbars(P)
    l=length(reals)
    println(l," real solutions found")
    r=rand(1:l);
    println("Plot solution no. ", r)
    oneSol=reals[r];
    plotFourbar(oneSol,P);
end

function example3()
    #3 points on a circle, 4 on another
    P=[Point2f0(0.25,0.00), Point2f0(0.52,0.10), Point2f0(0.80,0.70), Point2f0(1.20,1.00), Point2f0(1.40,1.30), Point2f0(1.10,1.48), Point2f0(0.70,1.40), Point2f0(0.20,1.00), Point2f0(0.02,0.40)].*10;
    println(P);
    reals=findRealFourbars(P)
    l=length(reals)
    println(l," real solutions found")
    r=rand(1:l);
    println("Plot solution no. ", r)
    oneSol=reals[r];
    plotFourbar(oneSol,P);
end

function example4()
    #nine points on an ellipse
    P=[Point2f0(1.000,0.875), Point2f0(0.750,0.625), Point2f0(0.500,0.375), Point2f0(0.250,0.125), Point2f0(0.000,0.000), Point2f0(0.96824583,1.32287565), Point2f0(1.56124949,1.73205080), Point2f0(1.85404962,1.93649167), Point2f0(1.98431348,2.00000000)].*5;
    println(P);
    reals=findRealFourbars(P)
    l=length(reals)
    println(l," real solutions found")
    r=rand(1:l);
    println("Plot solution no. ", r)
    oneSol=reals[r];
    plotFourbar(oneSol,P);
end
