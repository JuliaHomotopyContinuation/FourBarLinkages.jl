using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate();

using FourBarLinkages

# coupler points in isotropic coordinates
coupler_points = [complex(0.8961867,-0.09802917),
        complex(1.2156535, -1.18749100),
        complex(1.5151435, -0.85449808),
        complex(1.6754775,  -0.48768058),
        complex(1.7138690,-0.30099232),
        complex(1.7215236,0.03269953),
        complex(1.6642029, 0.33241088),
        complex(1.4984171, 0.74435576),
        complex(1.3011834,  0.92153806)]

# You can compute a generic solution and store it in a file with the following line:
#       compute_generic_solutions(; filename=data/four_bar_start_solutions.jld2)

# compute four bar linkages for the couple points from the stored results
fourbars = four_bars(coupler_points, "data/four_bar_start_solutions.jld2")

# pick a fourbar
F = fourbars[4]
# Δt is discretization step size for tracing out the curve
# compute all configuartions (angles) which trace out (a part of) the coupler curve
curves = configurations(F, coupler_points; Δt=1e-3)
# let's animate this with Makie
animate(F, curves[1], coupler_points)
# create endless loop (interrupt to stop)
animate(F, curves[1], coupler_points; loop=true)
# save animation and hide axis
FBL.animate(F, curves[1], coupler_points; filename="four-bar.gif", show_axis=false)
