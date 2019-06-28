import Pkg; Pkg.activate(@__DIR__);
using HomotopyContinuation
using FourBarLinkages


coupler_points = [complex(0.8961867,-0.09802917),
        complex(1.2156535, -1.18749100),
        complex(1.5151435, -0.85449808),
        complex(1.6754775,  -0.48768058),
        complex(1.7138690,-0.30099232),
        complex(1.7215236,0.03269953),
        complex(1.6642029, 0.33241088),
        complex(1.4984171, 0.74435576),
        complex(1.3011834,  0.92153806)]


gen_sol = load_generic_solutions("four_bar_start_solution.jld2")
fourbars = four_bars(coupler_points, "four_bar_start_solution.jld2")









res
