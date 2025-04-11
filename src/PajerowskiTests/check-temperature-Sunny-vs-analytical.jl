using Pkg
Pkg.activate("C:/Users/vdp/ORNL Dropbox/Daniel Pajerowski/Spiral/Sunny.jl")

# Test example: at T = 0 the δS correction for a 2D square-lattice antiferromagnet
# with and nearest-neighbor Heisenberg exchange is known to be δS ≈ 0.197
# (Huse, Phys. Rev. B 37, 2380 (1988)).

using Sunny, GLMakie


units = Units(:meV, :angstrom)

# Parameters
S = 1.0           # Spin
J = 1.0           # Exchange
D = -1e-3         # Anisotropy

function δS_SL_AFM(mode, kT)
    a = 1
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 90)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys = System(cryst, [1 => Moment(s=S, g=2)], mode;dims=(2, 2, 1))
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
    set_onsite_coupling!(sys, S -> D*S[3]^2, 1)
    set_dipole!(sys, (0, 0, +1), position_to_site(sys, (0, 0, 0)))
    set_dipole!(sys, (0, 0, -1), position_to_site(sys, (1, 0, 0)))
    set_dipole!(sys, (0, 0, -1), position_to_site(sys, (0, 1, 0)))
    set_dipole!(sys, (0, 0, +1), position_to_site(sys, (1, 1, 0)))
    swt = SpinWaveTheory(sys; measure=nothing)
    # Calculate first 3 digits for faster testing
    δS = Sunny.magnetization_lswt_correction(swt; kT=kT, atol=1e-3)[1]
    return δS
end

function δE_SL_AFM(mode, kT)
    a = 1
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 90)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys = System(cryst, [1 => Moment(s=S, g=2)], mode;dims=(2, 2, 1))
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
    set_onsite_coupling!(sys, S -> D*S[3]^2, 1)
    set_dipole!(sys, (0, 0, +1), position_to_site(sys, (0, 0, 0)))
    set_dipole!(sys, (0, 0, -1), position_to_site(sys, (1, 0, 0)))
    set_dipole!(sys, (0, 0, -1), position_to_site(sys, (0, 1, 0)))
    set_dipole!(sys, (0, 0, +1), position_to_site(sys, (1, 1, 0)))
    swt = SpinWaveTheory(sys; measure=nothing)
    δE = Sunny.energy_per_site_lswt_correction(swt; kT=kT, atol=1e-4)
    return δE
end


function analytical_δS_integrand(qx, qy, kT)
    eps = 0
    D_c = D * (1-1/2S)
    ω = (4 * J * S + 2 * S * abs(D_c)) * (sqrt(1 - (J*cos(qx) + J*cos(qy))^2/(2*J+abs(D_c))^2) + eps)
    nk = kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
    return (nk + 0.5) / (sqrt(1 - (J*cos(qx) + J*cos(qy))^2/(2*J+abs(D_c))^2) + eps)
end

function analytical_δS(kT)
    integral, _ = Sunny.hcubature(q -> analytical_δS_integrand(q[1], q[2], kT), (-π, -π), (π, π); atol=1e-3)
    δS = -( integral / (2π)^2 - 0.5)
    return δS
end

function analytical_δE_integrand(qx, qy, kT)
    eps = 0.0
    D_c = D * (1 - 1 / (2S))
    A = 4 * J * S + 2 * S * abs(D_c)
    #A = 4 * J * S + abs(D_c)
    B = (J * cos(qx) + J * cos(qy)) / (2J + abs(D_c))
    ω = A * (sqrt(1 - B^2) + eps)

    nk = kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
    return (0.5 + nk) * ω
end

function analytical_δE(kT)
    integral, _ = Sunny.hcubature(q -> analytical_δE_integrand(q[1], q[2], kT), (-π, -π), (π, π); atol=1e-4)
    δE₂ = integral / (2π)^2

    # Normal ordering correction
    D_c = D * (1 - 1 / (2S))
    #A = 2J*S + D_c*S
    A = 2 * J * S + 1 * S * abs(D_c)
    δE₁ = -A  # energy per site

    return δE₁ + δE₂
end



# Temperature range in Kelvin
Ts_K = range(0, stop=0.5*J / units.K, length=11)
kTs = Ts_K * units.K  # Convert to meV

δS_analytical = Float64[]
δS_Sunny_dipole = Float64[]
δS_Sunny_SUN = Float64[]

for kT in kTs
    println("kT=$kT meV")

    push!(δS_analytical, analytical_δS(kT) )
    push!(δS_Sunny_dipole, δS_SL_AFM(:dipole, kT) )
    push!(δS_Sunny_SUN, δS_SL_AFM(:SUN, kT) )
end

δE_analytical = Float64[]
δE_Sunny_dipole = Float64[]
δE_Sunny_SUN = Float64[]

for kT in kTs
    println("kT=$kT meV")

    push!(δE_analytical, analytical_δE(kT) )
    push!(δE_Sunny_dipole, δE_SL_AFM(:dipole, kT) )
    push!(δE_Sunny_SUN, δE_SL_AFM(:SUN, kT) )
end

# Plot δS vs Temperature
fig = Figure(size=(600, 600))

ax1 = Axis(fig[1, 1],
    title = "Magnetization Correction δS vs Temperature",
    xlabel = "Temperature (K)",
    ylabel = "δS",
    xtickformat = x -> string.(round.(x, digits=1)),
    ytickformat = y -> string.(round.(y, digits=3))    
)

lines!(ax1, Ts_K, δS_analytical, label = "Analytical", color = :blue)
scatter!(ax1, Ts_K, δS_Sunny_dipole, label = "Sunny :dipole", color = :red, marker = :circle)
scatter!(ax1, Ts_K, δS_Sunny_SUN, label = "Sunny :SUN", color = :green, marker = :utriangle)
hlines!(ax1, [-0.197], linestyle = :dash, color = :black, label = "Huse (T = 0)")

axislegend(ax1, position = :rt)

ax2 = Axis(fig[2, 1],
    title = "Energy Correction δE vs Temperature",
    xlabel = "Temperature (K)",
    ylabel = "δE",
    xtickformat = x -> string.(round.(x, digits=1)),
    ytickformat = y -> string.(round.(y, digits=3))    
)

lines!(ax2, Ts_K, δE_analytical, label = "Analytical", color = :blue)
scatter!(ax2, Ts_K, δE_Sunny_dipole, label = "Sunny :dipole", color = :red, marker = :circle)
scatter!(ax2, Ts_K, δE_Sunny_SUN, label = "Sunny :SUN", color = :green, marker = :utriangle)
hlines!(ax2, [-0.334], linestyle = :dash, color = :black, label = "Huse (T = 0)")
#−0.658JS^2

axislegend(ax2, position = :rb)

display(fig)
