using Pkg
Pkg.activate("C:/Users/vdp/ORNL Dropbox/Daniel Pajerowski/Spiral/Sunny.jl")

using Sunny, GLMakie


function energy_per_site_lswt_correction(sswt::SpinWaveTheorySpiral; opts...)
    any(in(keys(opts)), (:rtol, :atol, :maxevals)) || error("Must specify one of `rtol`, `atol`, or `maxevals` to control momentum-space integration.")

    (; swt, k, axis) = sswt
    (; sys, data) = swt
    Natoms = Sunny.natoms(sys.crystal)
    L = Sunny.nbands(swt)
    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    # Uniform correction at q = 0, branch 2 is the "center" one without k offset
    Sunny.swt_hamiltonian_dipole_spiral!(H, sswt, zero(Vec3); branch=2)
    δE₁ = -real(Sunny.tr(view(H, 1:L, 1:L))) / 2Natoms

    # Integrate zero-point energy over 3 spiral branches (q-k, q, q+k)
    δE₂ = Sunny.hcubature((0,0,0), (1,1,1); opts...) do q_reshaped
        total = 0.0
        for branch in 1:3
            Sunny.swt_hamiltonian_dipole_spiral!(H, sswt, q_reshaped; branch)
            ωs = Sunny.bogoliubov!(V, H)
            total += sum(view(ωs, 1:L))
        end
        return total / (2 * Natoms * 3)  # normalize over atoms and branches
    end

    return δE₁ + δE₂[1]
end

function magnetization_lswt_correction_dipole(sswt::SpinWaveTheorySpiral; opts...)
    L = Sunny.nbands(sswt.swt)
    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    δS = Sunny.hcubature((0,0,0), (1,1,1); opts...) do q
        reduction = zeros(Float64, L)
        for branch in 1:3
            Sunny.swt_hamiltonian_dipole_spiral!(H, sswt, Sunny.Vec3(q); branch)
            Sunny.bogoliubov!(V, H)
            for i in 1:L
                # Each site's correction is norm² of lower half of Bogoliubov vector
                reduction[i] += Sunny.norm2(view(V, L+i, 1:L))
            end
        end
        return Sunny.SVector{L}(-reduction[i] / 3 for i in 1:L)  # average over branches
    end

    return δS[1]  # discard error bars
end



units = Units(:meV, :angstrom)
latvecs = lattice_vectors(6, 6, 40, 90, 90, 120)
cryst = Crystal(latvecs, [[1/2, 0, 0]], 147)
view_crystal(cryst; ndims=2)

sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = 1.0
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 1))
set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 2))
set_dipole!(sys, [cos(2π/3), sin(2π/3), 0], (1, 1, 1, 3))

println("**********************************")

B_field = 50.0
println("B_field=$B_field")
set_field!(sys, [0, 0, B_field] * units.T) #apply a magnetic field along the z-direction to get to the limiting case of diagonal magnons: 0 T for ground, 50 T for almost polarized, and 100 T for fully polarized

axis = [0, 0, 1]
randomize_spins!(sys)
k_3d = minimize_spiral_energy!(sys, axis; maxiters=100_000_000, k_guess = [0.3,0.3,0]) #can get bad local minima at times, be careful!
#k_3d = minimize_spiral_energy!(sys, axis; k_guess = [1/3,1/3,0])
#k_3d = minimize_spiral_energy!(sys, axis)

#k = [k_3d[1],k_3d[2],0]

axis_unit = Sunny.normalize(Sunny.SVector{3}(axis))
k_parallel = Sunny.dot(k_3d, axis_unit) * axis_unit
k = k_3d - k_parallel

if abs(sys.dipoles[1][3]) > 0.99
    k = [0,0,0]
end

println("k=$k, k3d=$k_3d")

sys_enlarged = repeat_periodically_as_spiral(sys, (3, 3, 1); k, axis)

minimize_energy!(sys_enlarged)

println("spiral energy per site k2d = $(spiral_energy_per_site(sys; k, axis))")
println("spiral energy per site k3d= $(spiral_energy_per_site(sys; k=k_3d, axis))")
println("supercell energy per site = $(energy_per_site(sys_enlarged))")

# Path
qs = [[-1/2, 0, 0], [0, 0, 0], [1/2, 1/2, 0]]
path = q_space_path(cryst, qs, 400)

# Spin-waves
swt1 = SpinWaveTheory(sys_enlarged; measure=ssf_perp(sys_enlarged))
res1 = intensities_bands(swt1, path)

swt2 = SpinWaveTheorySpiral(sys; measure=ssf_perp(sys), k, axis)
res2 = intensities_bands(swt2, path)

#plots
fig = Figure(size = (1200, 1000))

# Top: spin plots (3D)
scene1 = LScene(fig[1, 1])
scene2 = LScene(fig[1, 2])

plot_spins!(scene1, sys_enlarged; ndims=2)
plot_spins!(scene2, sys; ndims=2)

# Bottom-left: Supercell intensities
plot_intensities!(fig[2, 1], res1; units, saturation=0.5,
    axisopts=(xlabel="Q", ylabel="Energy (meV)", title="Supercell"))

# Bottom-right: Spiral intensities
plot_intensities!(fig[2, 2], res2; units, saturation=0.5,
    axisopts=(xlabel="Q", ylabel="Energy (meV)", title="Spiral"))

display(fig)

@time energy_per_site_lswt_correction_spiral = energy_per_site_lswt_correction(swt2; atol = 1e-6)
@time energy_per_site_lswt_correction_supercell = Sunny.energy_per_site_lswt_correction(swt1; kT=0*units.K, atol = 1e-6)

@time magnetization_lswt_correction_dipole_spiral = magnetization_lswt_correction_dipole(swt2; atol = 1e-2)
@time magnetization_lswt_correction_dipole_supercell = Sunny.magnetization_lswt_correction_dipole(swt1; kT= 0*units.K, atol = 1e-1)

println("spiral lswt-correction energy per site = $energy_per_site_lswt_correction_spiral")
println("supercell lswt-correction energy per site = $energy_per_site_lswt_correction_supercell")

println("spiral lswt-correction magnetization = $magnetization_lswt_correction_dipole_spiral")
println("supercell lswt-correction magnetization = $magnetization_lswt_correction_dipole_supercell")


using TestItems


    J = 1
    s = 1
    δE_afm1_ref = 0.488056/(2s) * (-2*J*s^2)

# The results are taken from Phys. Rev. B 102, 220405(R) (2020) for the AFM1
# phase on the FCC lattice
function correction(mode)
    a = 1
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    positions = [[0, 0, 0]]
    fcc = Crystal(latvecs, positions, 225)
    sys_afm1 = System(fcc, [1 => Moment(; s, g=1)], mode)
    set_exchange!(sys_afm1, J, Bond(1, 2, [0, 0, 0]))
    set_dipole!(sys_afm1, (0, 0,  1), position_to_site(sys_afm1, (0, 0, 0)))
    set_dipole!(sys_afm1, (0, 0, -1), position_to_site(sys_afm1, (1/2, 1/2, 0)))
    set_dipole!(sys_afm1, (0, 0, -1), position_to_site(sys_afm1, (1/2, 0, 1/2)))
    set_dipole!(sys_afm1, (0, 0,  1), position_to_site(sys_afm1, (0, 1/2, 1/2)))
    swt_afm1 = SpinWaveTheory(sys_afm1; measure=nothing)
    # Calculate at low accuracy for faster testing
    δE_afm1 = Sunny.energy_per_site_lswt_correction(swt_afm1; atol=5e-4)
    return isapprox(δE_afm1_ref, δE_afm1; atol=1e-3)
end

for mode in (:dipole, :SUN)
    correction(mode)
end



# Test example 1: The magnetization is maximized to `s`. Reference result
# comes from Phys. Rev. B 79, 144416 (2009) Eq. (45) for the 120° order on
# the triangular lattice.
J = 1
s = 1/2
a = 1
δS_ref = -0.261302

function δS_triangular(mode)
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys = System(cryst, [1 => Moment(s=s, g=2)], mode)
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
    polarize_spins!(sys, [0, 1, 0])
    sys = repeat_periodically_as_spiral(sys, (3, 3, 1); k=[2/3, -1/3, 0], axis=[0, 0, 1])
    swt = SpinWaveTheory(sys; measure=nothing)
    # Calculate first 3 digits for faster testing
    δS = Sunny.magnetization_lswt_correction(swt; kT=units.K*5, atol=1e-3)[1]
    return δS
end

for mode in (:dipole, :SUN)
    println(δS_triangular(mode))
end


using Sunny, StaticArrays, Plots, LsqFit

# Set model parameters
J = 1
s = 1/2
a = 1
δS_ref = -0.261302  # Known zero-temp LSWT correction

# Model setup + LSWT correction function
function δS_triangular(mode, kT)
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 120)
    cryst = Crystal(latvecs, [[0, 0, 0]])
    sys = System(cryst, [1 => Moment(s=s, g=2)], mode)
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
    polarize_spins!(sys, [0, 1, 0])
    sys = repeat_periodically_as_spiral(sys, (3, 3, 1); k=[2/3, -1/3, 0], axis=[0, 0, 1])
    swt = SpinWaveTheory(sys; measure=nothing)
    δS = Sunny.magnetization_lswt_correction(swt; kT=kT, atol=1e-3)[1]
    return δS
end

# Evaluate δS at a few temperatures
Ts = [1e-6, 0.1, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0] .* units.K * 10
δSs = [δS_triangular(:dipole, T) for T in Ts]
ΔδSs = δSs .- δSs[1]  # thermal correction

# Fit ΔδS(T) = A * T^2
Tvals = Ts./units.K
T² = Tvals .^ 2
model(T², p) = p[1] .* T²
fit = LsqFit.curve_fit(model, T², ΔδSs, [0.0])
A = fit.param[1]
println("Fit: ΔδS(T) ≈ $A * T²")

# Plot
Plots.scatter(Tvals, ΔδSs; label="Data", xlabel="Temperature (K)", ylabel="ΔδS",
        title="Magnetization Correction vs T²", lw=2, legend=:topleft)
Plots.plot!(T -> A * T^2, minimum(Tvals):0.01:maximum(Tvals), label="Fit: A·T²", lw=2)



@show δS_triangular(:dipole)
@show δS_triangular(:SUN)

@testitem "LSWT correction to the ordered moments (s not maximized)" begin
    using LinearAlgebra
    # Test example 2: The magnetization is smaller than `s` due to easy-plane
    # single-ion anisotropy The results are derived in the Supplemental
    # Information (Note 12) of Nature Comm. 12.1 (2021): 5331.
    a = b = 8.3193
    c = 5.3348
    lat_vecs = lattice_vectors(a, b, c, 90, 90, 90)
    types = ["Fe"]
    positions = [[0, 0, 0]]
    cryst = Crystal(lat_vecs, positions, 113; types)

    s = 1
    J₁  = 0.266
    J₁′ = 0.1J₁
    Δ = Δ′ = 1/3
    D = 1.42
    gab, gcc = 2.18, 1.93
    g = diagm([gab, gab, gcc])
    x = 1/2 - D/(8*(2J₁+J₁′))

    sys = System(cryst, [1 => Moment(; s, g)], :SUN; dims=(1, 1, 2), seed=0)
    set_exchange!(sys, diagm([J₁, J₁, J₁*Δ]),  Bond(1, 2, [0, 0, 0]))
    set_exchange!(sys, diagm([J₁′, J₁′, J₁′*Δ′]), Bond(1, 1, [0, 0, 1]))
    set_onsite_coupling!(sys, S -> D*S[3]^2, 1)

    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=1000)
    swt = SpinWaveTheory(sys; measure=nothing)

    δS = Sunny.magnetization_lswt_correction(swt; atol=1e-2)[1]

    M_cl  = 2*√((1-x)*x)
    # Paper reported M_ref = 2.79, but actual result is closer to 2.78
    M_ref = 2.78
    @test isapprox(M_ref, (M_cl+δS)*√3*gab, atol=1e-2)
end