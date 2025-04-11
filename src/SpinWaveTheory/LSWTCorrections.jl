"""
    energy_per_site_lswt_correction(swt::SpinWaveTheory; kT=0.0, opts...)

Computes the [𝒪(1/λ) or 𝒪(1/S)] correction to the classical energy **per
site** [𝒪(λ²) or 𝒪(S²)] given a [`SpinWaveTheory`](@ref). The correction
[𝒪(λ) or 𝒪(S)] includes a uniform term (For instance, if the classical energy
is αJS², the LSWT gives a correction like αJS) and the summation over the
zero-point energy for all spin-wave modes, i.e., 1/2 ∑ₙ ∫d³q ω(q, n), where q
belongs to the first magnetic Brillouin zone and n is the band index.  If a
nonzero `kT` is provided, the zero-point energy of the spin-wave modes is thermally
corrected via the Bose factor, i.e., (1/2 + n_B(ωₙ, kT)) * ωₙ.

A keyword argument `rtol`, `atol`, or `maxevals` is required to control the
accuracy of momentum-space integration. See the HCubature package documentation
for details.
"""
function energy_per_site_lswt_correction(swt::SpinWaveTheory; kT=0.0, opts...)
    any(in(keys(opts)), (:rtol, :atol, :maxevals)) || error("Must specify one of `rtol`, `atol`, or `maxevals` to control momentum-space integration.")

    (; sys) = swt
    Natoms = natoms(sys.crystal)
    L = nbands(swt)
    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    # The uniform correction at q=0 to the classical energy (trace of the
    # (1,1)-block of the spin-wave Hamiltonian)
    dynamical_matrix!(H, swt, zero(Vec3))
    δE₁ = -real(tr(view(H, 1:L, 1:L))) / 2Natoms

    # Quantum or thermal correction from dispersion
    δE₂ = _integrate_lswt_energy_correction_qspace(swt; kT, opts...)

    # Error bars in δE₂[2] are discarded
    @show δE₁, δE₂
    return δE₁ + δE₂
end

# Internal helper for Brillouin-zone integration of LSWT energy correction
function _integrate_lswt_energy_correction_qspace(swt::SpinWaveTheory; kT::Real=0.0, opts...)
    (; sys) = swt
    Natoms = natoms(sys.crystal)
    L = nbands(swt)
    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    # Integrate zero-point energy over the first Brillouin zone 𝐪 ∈ [0, 1]³ for
    # magnetic cell in reshaped RLU
    return hcubature((0, 0, 0), (1, 1, 1); opts...) do q_reshaped
        dynamical_matrix!(H, swt, q_reshaped)
        ωs = bogoliubov!(V, H)
        if kT == 0
            return sum(view(ωs, 1:L)) / 2Natoms
        else
            return sum(ω -> (0.5 + 1 / (exp(ω / kT) - 1)) * ω, view(ωs, 1:L)) / Natoms
        end
    end |> first  # discard estimated error bar
end

# Calculates the magnetization reduction for :SUN mode for all atoms
function magnetization_lswt_correction_sun(swt::SpinWaveTheory; kT::Real = 0.0, opts...)
    (; sys, data) = swt

    N = sys.Ns[1]
    Natoms = natoms(sys.crystal)
    L = (N - 1) * Natoms

    H = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    # Construct angular momentum operators O = n⋅S aligned with quantization axis
    S = spin_matrices_of_dim(; N)
    O = zeros(ComplexF64, N, N, Natoms)
    for i in 1:Natoms
        n = normalize(sys.dipoles[i])
        U = data.local_unitaries[i]
        O[:, :, i] .= U' * (n' * S) * U
        @assert O[N, N, i] ≈ norm(sys.dipoles[i])
    end

    integrand = if kT == 0
        q -> begin
            swt_hamiltonian_SUN!(H, swt, q)
            ωs = bogoliubov!(V, H)
            #@show minimum(ωs), maximum(ωs)
            out = zeros(Natoms)
            for band in L+1:2L
                v = reshape(view(V, :, band), N-1, Natoms, 2)
                for i in 1:Natoms, α in 1:N-1, β in 1:N-1
                    out[i] -= real((O[N, N, i] * δ(α, β) - O[α, β, i]) * conj(v[α, i, 1]) * v[β, i, 1])
                end
            end
            return SVector{Natoms}(out)
        end
    else
        q -> begin
            
            swt_hamiltonian_SUN!(H, swt, q)
            ωs = bogoliubov!(V, H)
            #@show minimum(ωs), maximum(ωs)
            out = zeros(Natoms)
            for (b, band) in enumerate(L+1:2L)
                v = reshape(view(V, :, band), N-1, Natoms, 2)
                #ω = ωs[b]  # positive eigenvalue
                ω = ωs[band - L]  # positive ω
                n_th = 1 / (exp(ω / kT) - 1)
                #@show ω, n_th
                factor = 1 + 2 * n_th
                for i in 1:Natoms, α in 1:N-1, β in 1:N-1
                    out[i] -= factor * real((O[N, N, i] * δ(α, β) - O[α, β, i]) * conj(v[α, i, 1]) * v[β, i, 1])
                end
            end
            return SVector{Natoms}(out)
        end
    end

    δS, _ = hcubature(integrand, (0, 0, 0), (1, 1, 1); opts...)
    return δS
end



# Calculates the magnetization reduction for :dipole mode for every site
function magnetization_lswt_correction_dipole(swt::SpinWaveTheory; kT::Real = 0.0, opts...)
    L = nbands(swt)
    Natoms = Sunny.natoms(swt.sys.crystal)
    N = 2#swt.sys.Ns[1]  # For dipole mode, N should be 2, Ns is used to keep track of S for renorms
    H = zeros(ComplexF64, 2L, 2L)
    V = similar(H)

    integrand = if kT == 0
        q -> begin
            swt_hamiltonian_dipole!(H, swt, Vec3(q))
            bogoliubov!(V, H)
            out = zeros(Natoms)
            for band in L+1:2L
                v = reshape(view(V, :, band), N-1, Natoms, 2)
                for i in 1:Natoms
                    out[i] -= real(conj(v[1, i, 1]) * v[1, i, 1])
                end
            end
            return SVector{Natoms}(out)
        end
    else
        q -> begin
            swt_hamiltonian_dipole!(H, swt, Vec3(q))
            ωs = bogoliubov!(V, H)
            out = zeros(Natoms)
            for (b, band) in enumerate(L+1:2L)
                v = reshape(view(V, :, band), N-1, Natoms, 2)
                ω = ωs[b]
                n_th = 1 / (exp(ω / kT) - 1)
                factor = 1 + 2 * n_th
                for i in 1:Natoms
                    out[i] -= factor * real(conj(v[1, i, 1]) * v[1, i, 1])
                end
            end
            return SVector{Natoms}(out)
        end
    end

    δS, _ = hcubature(integrand, (0, 0, 0), (1, 1, 1); opts...)
    return δS
end



"""
    magnetization_lswt_correction(swt::SpinWaveTheory; kT=0.0, opts...)

Calculates the reduction in the classical dipole magnitude for all atoms in the
magnetic cell. In the case of `:dipole` and `:dipole_uncorrected` mode, the
classical dipole magnitude is constrained to spin-`s`. While in `:SUN` mode, the
classical dipole magnitude can be smaller than `s` due to anisotropic
interactions. The optional `kT` parameter adds thermal corrections to the LSWT
fluctuations.

A keyword argument `rtol`, `atol`, or `maxevals` is required to control the
accuracy of momentum-space integration. See the HCubature package documentation
for details.
"""
function magnetization_lswt_correction(swt::SpinWaveTheory; opts...)
    any(in(keys(opts)), (:rtol, :atol, :maxevals)) || error("Must specify one of `rtol`, `atol`, or `maxevals` to control momentum-space integration.")

    (; sys) = swt
    if sys.mode == :SUN
        δS = magnetization_lswt_correction_sun(swt; opts...)
    else
        @assert sys.mode in (:dipole, :dipole_uncorrected)
        δS = magnetization_lswt_correction_dipole(swt; opts...)
    end
    return δS
end
