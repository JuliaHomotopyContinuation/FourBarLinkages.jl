
struct LinkageData
    A::ComplexF64
    B::ComplexF64
    r₂::Float64
    r₃::Float64
    r₄::Float64
end

function angles(L::LinkageData, θ₂, first_branch::Bool)
    f = L.B - L.A
    R₁ = L.r₃
    R₂ = -L.r₄
    z = f - L.r₂ * cis(θ₂)
    x₃, y₃ = reim(z)
    branch = acos((x₃^2 + y₃^2 + R₂^2 - R₁^2) / (2R₂ * abs(z)))
    θ₄ = atan(y₃, x₃)
    if first_branch
        θ₄ += branch
    else
        θ₄ -= branch
    end
    θ₃ = atan((y₃ - R₂ * sin(θ₄)) / R₁, (x₃ - R₂ * cos(θ₄)) / R₁)
    θ₄, θ₃
end
Base.broadcastable(L::LinkageData) = Ref(L)

L = LinkageData(0. + 0im, 2. + 0im, 1, 2., 1)
α = range(0.0, 2π, length=10)
# this should result in (α, 0)
angles.(L, α, true)
