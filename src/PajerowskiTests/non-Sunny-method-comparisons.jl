using LinearAlgebra
using GLMakie
using HCubature

# Model parameters
J = 1.0                        # Exchange in meV
D = 1e-7                     #uniaxiall anisotropy
S = 1.0                        # Spin
kB_meV_per_K = 1 / 11.6       # meV/K


# Integration grid
function gamma(qx, qy)
    0.5 * (cos(qx) + cos(qy))
end



function dS_lecture_formula(qx, qy, kT)
    eps = 1e-6
    γk = gamma(qx, qy)
    ω = 4 * J * S * (sqrt(1 - γk^2) + eps)
    nk = kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
    return (nk + 0.5) / (sqrt(1 - γk^2) + eps)
end

function ω_analytical(qx, qy)
    A = 2 * J * S + D * S
    B = 2 * J * S * gamma(qx, qy)
    sqrt(A^2 - B^2)
end

function v_squared_analytical(qx, qy)
    A = 2 * J * S + D * S
    B = 2 * J * S * gamma(qx, qy)

    ω = sqrt(A^2 - B^2)

    v2 = (1/2)*(A/ω-1)

    return v2
end

function v_squared_bdg(qx, qy)
    A = 2 * J * S + D * S
    B = 2 * J * S * gamma(qx, qy)
    H_BdG = [A B; -B -A]

    evals, evecs = eigen(H_BdG)

    v_sum = 0.0
    sigma_z = Diagonal([1.0, -1.0])

    for idx in 1:2
        ω = real(evals[idx])
        vec = evecs[:, idx]

        norm_val = vec' * sigma_z * vec
        if real(norm_val) > 0
            vec ./= sqrt(abs(norm_val))
            v_sum += abs2(vec[2])
        end
    end

    return v_sum
end


function ω_bdg(qx, qy)
    A = 2 * J * S + D * S
    B = 2 * J * S * gamma(qx, qy)
    H_BdG = [A B; -B -A]
    evals = eigen(H_BdG).values
    maximum(real(evals))
end

function bose_factor(ω, kT)
    kT ≈ 0.0 ? 0.0 : 1 / (exp(ω / kT) - 1)
end

function energy_integrand(qx, qy, ωfunc, kT)
    ω = ωfunc(qx, qy)
    (0.5 + bose_factor(ω, kT)) * ω
end

function magnetization_integrand(qx, qy, ωfunc, v2func, kT)
    ω = ωfunc(qx, qy)
    factor = 1 + 2*bose_factor(ω, kT)
    return factor * v2func(qx, qy)
end

"""
    δS_sum_method(f::Function, kT; Nq=200)

Estimate δS using a regular Brillouin zone grid summation.
Takes a function f(qx, qy, kT) representing the δS integrand.
"""
function δS_sum_method(f::Function, kT; Nq=100)
    eps = 1e-20
    qx_vals = range(-π, π; length=Nq)
    qy_vals = range(-π, π; length=Nq)
    #qx_vals = range(-π, π; length=Nq+1)[1:end-1]
    #qy_vals = range(-π, π; length=Nq+1)[1:end-1]
    
    dq² = (2π / Nq)^2

    sum_val = 0.0
    for qx in qx_vals, qy in qy_vals
        γ = gamma(qx, qy)
        if 1 - γ^2 < eps
            continue  # skip singular Gamma point
        end
        sum_val += f(qx, qy, kT)
    end
    return sum_val * dq²
end


# Temperature range in Kelvin
Ts_K = range(0, stop=0.5 * 11.6, length=21)  # Up to 2J
kTs = kB_meV_per_K .* Ts_K  # Convert to meV

# Storage
δEs_analytical = Float64[]
δSs_analytical = Float64[]
δEs_bdg = Float64[]
δSs_bdg = Float64[]

δSs_lecture = Float64[]

for kT in kTs
    println("kT=$kT")
    δE_a, _ = hcubature(q -> energy_integrand(q[1], q[2], ω_analytical, kT), (-π, -π), (π, π); rtol=1e-4, atol=1e-4)
    δS_a, _ = hcubature(q -> magnetization_integrand(q[1], q[2], ω_analytical, v_squared_analytical, kT), (-π, -π), (π, π); rtol=1e-4, atol=1e-4)
    δE_b, _ = hcubature(q -> energy_integrand(q[1], q[2], ω_bdg, kT), (-π, -π), (π, π); rtol=1e-4, atol=1e-4)
    δS_b, _ = hcubature(q -> magnetization_integrand(q[1], q[2], ω_bdg, v_squared_bdg, kT), (-π, -π), (π, π); rtol=1e-4, atol=1e-4)

    push!(δEs_analytical, δE_a / (2π)^2)
    push!(δSs_analytical, δS_a / (2π)^2)
    push!(δEs_bdg, δE_b / (2π)^2)
    push!(δSs_bdg, δS_b / (2π)^2)

    δS_c, _ = hcubature(q -> dS_lecture_formula(q[1], q[2], kT), (-π, -π), (π, π); rtol=1e-4, atol=1e-4)
    push!(δSs_lecture, ( δS_c / (2π)^2 - 0.5)  )

end

calc_sum = false
if calc_sum == true
    δSs_analytical_sum = Float64[]
    δSs_bdg_sum = Float64[]
    δSs_lecture_sum = Float64[]

    for kT in kTs
        println("kT=$kT")
        push!(δSs_analytical_sum, δS_sum_method((qx, qy, kT) -> magnetization_integrand(qx, qy, ω_analytical, v_squared_analytical, kT), kT))
        push!(δSs_bdg_sum,       δS_sum_method((qx, qy, kT) -> magnetization_integrand(qx, qy, ω_bdg,       v_squared_bdg,       kT), kT))
        push!(δSs_lecture_sum,   δS_sum_method(dS_lecture_formula, kT)  - 0.5)  # Subtract 0.5 for lecture style
    end
end

# Plot
fig = Figure(size=(800, 600))

ax1 = Axis(fig[1, 1], title="Energy Correction vs Temperature", xlabel="Temperature (K)", ylabel="ΔE per site (meV)")
lines!(ax1, Ts_K, δEs_analytical, label="Analytical")
lines!(ax1, Ts_K, δEs_bdg, label="BdG")
axislegend(ax1)

ax2 = Axis(fig[2, 1], title="Magnetization Correction vs Temperature", xlabel="Temperature (K)", ylabel="δS")
scatter!(ax2, Ts_K, δSs_analytical, label="Analytical")
scatter!(ax2, Ts_K, δSs_bdg, label="BdG")
scatter!(ax2, Ts_K, δSs_lecture, label="Lecture Notes", color=:purple)
hlines!(ax2, [0.197], color=:black, linestyle=:dash, label="Huse (T=0)")


#lines!(ax2, Ts_K, δSs_analytical_sum, label="Analytical (Sum)", color=:green, linestyle=:dot)
#lines!(ax2, Ts_K, δSs_bdg_sum,       label="BdG (Sum)",       color=:orange, linestyle=:dot)
#lines!(ax2, Ts_K, δSs_lecture_sum,   label="Lecture Notes (Sum)", color=:purple, linestyle=:dashdot)


axislegend(ax2)

display(fig)