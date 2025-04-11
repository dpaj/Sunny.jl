using Pkg
Pkg.activate("C:/Users/vdp/ORNL Dropbox/Daniel Pajerowski/Spiral/Sunny.jl")

using Sunny, LinearAlgebra, GLMakie

latvecs = [
    1 0 0
    0 2 0
    0 0 3
]
positions = [[0, 0, 0], [0, 1/3, 0]]
crystal = Crystal(latvecs, positions)
view_crystal(crystal)

# We are interested in a quasi-1D system, without periodic boundary conditions
# along the b axis. For this reason, we set the second site slightly off of 
# the `[0, 0.5, 0]` position, which would induce Sunny to identify the bonds
# between the two sides (1 -> 2 and 2 -> 1) as equivalent.

# Specify a system and the two exchange interactions, J (rungs) and J′
# (lengthwise bonds). 

J = 1
J′ = 0.2J*1*0
sys = System(crystal, [1 => Moment(s=1/2, g=2)], :SUN; dims=(2, 1, 1))
set_exchange!(sys, J′, Bond(1, 1, [1, 0, 0]))
set_exchange!(sys, J, Bond(1, 2, [0, 0, 0]))

# Examine the behavior of this system when we randomize the spins and minimize
# the energy.

randomize_spins!(sys)
minimize_energy!(sys)
plot_spins(sys)

# ## Spin Wave Calculations

# The result is the expected classical ordering, with full-length dipoles
# arranged antiferromagnetically. This results in a `q=(π,π)` ordering. Because
# this model is Heisenberg, the Hamiltonian has an SU(2) symmetry, and any such
# ground state breaks this symmetry. This will lead to a Goldstone mode, as can
# readily be seen when we calculate the excitations with spin wave theory.

swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
qs = q_space_path(crystal, [[0, 1, 0], [1/2, 1, 0], [1, 1, 0]], 200)
energies = range(0, 2.5, 200)
res = intensities(swt, qs; energies, kernel=gaussian(; fwhm=0.2))
plot_intensities(res)

# This result is incorrect for such a small J′. Instead, the ground state on
# each bond should be a singlet, i.e. a non-magnetic ground state.
# Correspondingly, the excitations should be gapped singlet-triplet excitations.
# This can be reproduced using the entangled units formalism. An
# `EntangledSystem` is constructed from an ordinary `System` by providing a list
# of sites "to entangle" within each unit cell.

esys = Sunny.EntangledSystem(sys, [(1, 2)])
randomize_spins!(esys)
minimize_energy!(esys)
plot_spins(esys)

# The ground state here is a pair of singlets, and the magnitude is of the
# dipoles on each site is zero (up to numerical precision). The ordering wave
# vector is now `q=0`, so the system should be reshaped into the magnetic unit
# cell (one bond) before performing spin wave calculations.

identity3 = Matrix{Float64}(I, 3, 3)
esys = reshape_supercell(esys, identity3)

plot_spins(esys)

# Next create a `SpinWaveTheory` and calculate intensities just as would be
# done for an ordinary Sunny `System`.

eswt = SpinWaveTheory(esys; measure=ssf_trace(esys)) ## TODO: ssf functions for esys
res = intensities(eswt, qs; energies, kernel=gaussian(; fwhm=0.2))
plot_intensities(res)

#Sunny.energy_per_site_lswt_correction(eswt; atol = 1e-4)
#Sunny.energy_per_site_lswt_correction(swt; atol = 1e-4)


println("J = $J, J′ = $J′")

function analytical_ladder_δE(kT)
    # Dispersion part δE₂
    integral, _ = Sunny.hcubature(q -> begin
        A = J + J′ * cos(q[1])
        B = J′ * cos(q[1])
        ω = sqrt(A^2 - B^2)
        nk = kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
        (0.5 + nk) * ω
    end, (-π,), (π,); atol=1e-4)
    δE₂ = 3 * integral / (2π)

    # Normal ordering correction δE₁
    integral_dE1, _ = Sunny.hcubature(q -> begin
        A = J + J′ * cos(q[1])
        #B = J′ * cos(q[1])
        #ω = sqrt(A^2 - B^2)
        -3 * A/2
    end, (-π,), (π,); atol=1e-4)
    δE₁ = integral_dE1 / (2π)

    println("δE₁ = $δE₁")
    println("δE₂ = $δE₂")

    return δE₁ + δE₂
end

function analytical_chain_δE(kT)
    S = 1/2
    eps = 1e-7*0
    D_c = 1e-5*1
    # Dispersion part δE₂
    integral, _ = Sunny.hcubature(q -> begin
        A = J + abs(D_c)
        B = J * cos(q[1]) 
        ω = 2*S*sqrt(A^2 - B^2)
        #ω = A * (sqrt(1 - B^2) + eps)
        nk = kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
        (0.5 + nk) * ω
    end, (-π,), (π,); atol=1e-5, rtol = 1e-5)
    δE₂ = integral / (2π)

    # Normal ordering correction δE₁
    integral_dE1, _ = Sunny.hcubature(q -> begin
        A = J + abs(D_c)
        B = J * cos(q[1]) 
        ω = 2*S*sqrt(A^2 - B^2)
        #ω = sqrt(A^2 - B^2)
        #ω = A * (sqrt(1 - B^2) + eps)
        -A/4
    end, (-π,), (π,); atol=1e-5, rtol = 1e-5)
    δE₁ = integral_dE1 / (2π)

    println("δE₁ = $δE₁")
    println("δE₂ = $δE₂")

    return δE₁ + δE₂
end

kT = 0
println("analytical ladder correction = $(analytical_ladder_δE(kT))")

#println("analytical chain correction = $(analytical_chain_δE(kT))")

println("sunny entangled correction = $(Sunny.energy_per_site_lswt_correction(eswt; kT = kT, atol = 1e-4))")

#println("sunny unentangled correction = $(Sunny.energy_per_site_lswt_correction(swt; kT = kT, atol = 1e-4))")


Sunny.magnetization_lswt_correction(eswt; atol = 1e-4)

Natoms = Sunny.natoms(eswt.sys.crystal)
L = Sunny.nbands(eswt)
H = Sunny.zeros(ComplexF64, 2L, 2L)
V = Sunny.zeros(ComplexF64, 2L, 2L)

# The uniform correction at q=0 to the classical energy (trace of the
# (1,1)-block of the spin-wave Hamiltonian)
Sunny.dynamical_matrix!(H, eswt, Sunny.zero(Sunny.Vec3))

-real(tr(view(H, 1:L, 1:L))) / 2Natoms