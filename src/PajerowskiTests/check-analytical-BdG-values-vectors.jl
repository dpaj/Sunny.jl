using LinearAlgebra
using GLMakie

# Model parameters
J = 1.0
D = 0.0    # No anisotropy
S = 1.0

# High-symmetry path: Gamma (0,0) -> X (pi,0) -> M (pi,pi) -> Gamma (0,0)
path_q = vcat(
    [(x, 0.0) for x in range(0, stop=pi, length=50)],
    [(pi, y) for y in range(0, stop=pi, length=50)],
    [(x, x) for x in range(pi, stop=0, length=50)]
)

labels = ["Γ", "X", "M", "Γ"]
n = length(path_q)
ticks = [0, round(Int, n / 3), round(Int, 2n / 3), n]

# Functions
gamma(qx, qy) = 0.5 * (cos(qx) + cos(qy))

# Storage
angles = Float64[]
energies_analytical = Float64[]
energies_bdg = Float64[]
energy_diff = Float64[]
vec_analytical_components = Tuple{Float64, Float64}[]
vec_bdg_components = Tuple{Float64, Float64}[]

for (qx, qy) in path_q
    local γ = gamma(qx, qy)
    local A = 2 * J * S + D * S
    local B = 2 * J * S * γ

    # Analytical eigenvector and energy
    ω = sqrt(A^2 - B^2)
    vec_analytical = if abs(B) < 1e-12
        [1.0, 0.0]
    else
        v = (ω - A) / B
        nrm = sqrt(1 + abs2(v))
        [1.0, v] / nrm
    end

    # BdG Hamiltonian
    H_BdG = [A B; -B -A]
    evals, evecs = eigen(H_BdG)
    idx = argmax(real(evals))
    vec_bdg = evecs[:, idx]

    # Normalize and compare
    vec_analytical ./= norm(vec_analytical)
    vec_bdg ./= norm(vec_bdg)
    dot_product = dot(vec_analytical, vec_bdg)
    angle = acos(clamp(real(dot_product), -1, 1)) * 180 / π

    push!(angles, angle)
    push!(energies_analytical, ω)
    push!(energies_bdg, real(evals[idx]))
    push!(energy_diff, abs(ω - real(evals[idx])))
    push!(vec_analytical_components, (real(vec_analytical[1]), real(vec_analytical[2])))
    push!(vec_bdg_components, (real(vec_bdg[1]), real(vec_bdg[2])))
end

# Plot
fig = Figure(size=(800, 800))

ax1 = Axis(fig[1, 1], title="Eigenvector angle and components", xlabel="q-path", ylabel="Angle (°)", xticks=(ticks, labels))
lines!(ax1, x, angles, label="Angle (deg)")

lines!(ax1, x, vec_ana_1, label="Analytical v₁", color=:green, linestyle=:dash)
scatter!(ax1, x, vec_ana_1, color=:green, marker=:circle, markersize=14)

lines!(ax1, x, vec_ana_2, label="Analytical v₂", color=:green, linestyle=:dot)
scatter!(ax1, x, vec_ana_2, color=:green, marker=:circle, markersize=14)

lines!(ax1, x, vec_bdg_1, label="BdG v₁", color=:orange, linestyle=:dash)
scatter!(ax1, x, vec_bdg_1, color=:orange, marker=:cross, markersize=7)

lines!(ax1, x, vec_bdg_2, label="BdG v₂", color=:orange, linestyle=:dot)
scatter!(ax1, x, vec_bdg_2, color=:orange, marker=:cross, markersize=7)

axislegend(ax1)

ax2 = Axis(fig[2, 1], title="Magnon energies and difference", xlabel="q-path", ylabel="Energy", xticks=(ticks, labels))
lines!(ax2, x, energies_analytical, label="Analytical")
lines!(ax2, x, energies_bdg, label="BdG")
lines!(ax2, x, energy_diff, label="|ΔE|", color=:red, linestyle=:dashdot)
axislegend(ax2)

display(fig)
