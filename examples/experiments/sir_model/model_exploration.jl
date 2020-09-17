### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 70d9471e-f361-11ea-3d79-857e54fd86ec
begin
	import Pkg; Pkg.activate(".")
	using DifferentialEquations
	using DiffEqSensitivity
	using Random
	using Distributions
	using DataFrames
	using StatsPlots
	using Turing
	using PlutoUI
	using Quadrature
	using StatsFuns
end

# ╔═╡ 87b490e0-f363-11ea-3598-f772c67da114
Random.seed!(1234)

# ╔═╡ bf72c778-f362-11ea-1763-f1c7c2b761fb
function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,γ) = p
    N = S+I+R
    infection = β*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end

# ╔═╡ 851de356-f366-11ea-3bfa-6f28b87378bf
function base_reproduction_rate(β, γ)
    return β / γ
end

# ╔═╡ 97441d98-f50d-11ea-364b-0b9955b8afff
function predict(y,chain)
    # Length of data
    l = length(y)
    # Length of chain
    m = length(chain)
    # Choose random
    idx = sample(1:m)
    i₀ = Array(chain[:i₀])[idx]
    β = Array(chain[:β])[idx]
    I = i₀*10.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,0.25]
    tspan = (0.0,float(l))
    prob = ODEProblem(sir_ode!,
            u0,
            tspan,
            p)
    sol = solve(prob,
                Tsit5(),
                saveat = 1.0)
    out = Array(sol)
    sol_X = [0.0; out[4,2:end] - out[4,1:(end-1)]]
    return hcat(sol.t, out', sol_X)
end

# ╔═╡ 8acada86-f50d-11ea-2979-2f821a08aee0
function plot_predictive(obstimes, X_true, Y_obs, chain)
    Xp = []
    for i in 1:20
        pred = predict(Y_obs, chain)
        push!(Xp, pred[2:end,6])
    end

    p = plot(obstimes, Xp; legend=false, color=:red, alpha=0.8)
    plot!(p, obstimes, X_true, color=:black, lw=3)
    scatter!(p, obstimes, Y_obs)
    return p
end

# ╔═╡ 43713eae-f36f-11ea-07e5-3143c8a802cd
function gen_logjoint(v, model, spl)
    function logjoint(z)::Float64
        z_old, lj_old = v[spl], Turing.DynamicPPL.getlogp(v) 
        v[spl] = z
        model(v, spl)
        lj = Turing.DynamicPPL.getlogp(v)
        v[spl] = z_old
        Turing.DynamicPPL.setlogp!(v, lj_old)
        return lj
    end
    return logjoint
end

# ╔═╡ 4f3a5b1a-f59c-11ea-2121-4bff69f69ddc
function gen_logprior(v, model, spl)
    function logprior(z)::Float64
        z_old, lj_old = v[spl], Turing.DynamicPPL.getlogp(v)
        v[spl] = z
        model(v, Turing.SampleFromPrior(), Turing.DynamicPPL.PriorContext())
        lj = Turing.DynamicPPL.getlogp(v)
        v[spl] = z_old
        Turing.DynamicPPL.setlogp!(v, lj_old)
        return lj
    end
    return logprior
end

# ╔═╡ beb376b6-f362-11ea-3df6-611ae46edd16
tmax = 15.0;

# ╔═╡ 40c3a9c8-f363-11ea-3056-0172f86c11f3
tspan = (0.0,tmax);

# ╔═╡ 5020c414-f363-11ea-3d8d-3d124c684663
obstimes = 1.0:1.0:tmax;

# ╔═╡ aa483766-f367-11ea-21e8-a3094913df8c
total_population = 10_000;

# ╔═╡ b6505484-f363-11ea-39c2-cf47a29fd04c
@bind i0 Slider(0:20)

# ╔═╡ 98532414-f365-11ea-05e7-f3adfa263d75
md"""
Initial conditions:

Susceptible: $(total_population - 10*i0)

Infected: $(10*i0)

Recovered: 0
"""

# ╔═╡ 4fae7a58-f363-11ea-2f53-e3e7406a27e4
begin
	I0 = i0 * 10
	S0 = total_population - I0
	u0 = [S0, I0, 0.0, 0.0] # S,I.R,C
end;

# ╔═╡ 09830f94-f368-11ea-0356-d7bb1bec4702
@bind beta Slider(0.1:0.01:2.0)

# ╔═╡ f9443cd0-f366-11ea-11db-152aeab61686
md"Beta: $beta"

# ╔═╡ 4f745940-f363-11ea-136d-cb3d8ef9dec2
begin
	gamma = 0.25
	p = [beta,gamma]; # β,c,γ
end;

# ╔═╡ a731c82c-f366-11ea-06c1-d77b42a7ad86
md"R0: $(base_reproduction_rate(p...))"

# ╔═╡ 5d23f438-f363-11ea-03b6-01c3c4a5b784
prob_ode = ODEProblem(sir_ode!,u0,tspan,p);

# ╔═╡ 22f63156-f366-11ea-33d6-193ab9db59a9
sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0);

# ╔═╡ 7883dd38-f363-11ea-2dba-219d4ecc794d
C = Array(sol_ode)[4,:];

# ╔═╡ 7a9b89fe-f363-11ea-370a-75afb531a410
X = C[2:end] - C[1:(end-1)];

# ╔═╡ 8cc14074-f363-11ea-2523-13016e95943b
#Y = rand.(Poisson.(X));

# ╔═╡ 302e99b0-f366-11ea-3110-43325fee9342
md"""
Final conditions:

Total population: 10000

Susceptible: $(Array(sol_ode)[1,end])

Infected: $(Array(sol_ode)[2,end])

Recovered: $(Array(sol_ode)[3,end])

Cumulative cases: $(Array(sol_ode)[3,end])
"""

# ╔═╡ 07fc5356-f50a-11ea-0ef1-2dbf0cb60038
neg_bin_r(mean, var) = mean^2 / (var - mean)

# ╔═╡ 2ae8e6f6-f50a-11ea-02b5-19ddd496b7aa
neg_bin_p(r, mean) = r / (r + mean)

# ╔═╡ 52d0d8ae-f50a-11ea-0939-d95feb97d248
function neg_bin_params(mean, var)
	r = neg_bin_r(mean, var)
	return r, neg_bin_p(r, mean)
end

# ╔═╡ e24e1b1a-f5d0-11ea-10dc-b5e3ba2652ae
neg_bin_params2(mean, phi) = neg_bin_params(mean, mean + mean^2 / phi)

# ╔═╡ 666ee530-f50b-11ea-00aa-eb40e3f3cdd3
Y = rand.([NegativeBinomial(neg_bin_params2(m, 10)...) for m in X]);

# ╔═╡ 920a63c6-f363-11ea-3c99-a97f3db6d4c0
begin
	bar(
		obstimes,
		Y,
		legend=false
	)
	plot!(obstimes,X,legend=false)
end

# ╔═╡ d3a7154c-f36b-11ea-21f6-1569109c69eb
@model bayes_sir(y, total_population) = begin
	# Calculate number of timepoints
	l = length(y)
	i₀ ~ truncated(Normal(10, 10), 0, total_population/10)
	β ~ truncated(Normal(1, 0.5), 0, Inf)
	I = i₀ * 10
	u0=[total_population-I, I, 0.0, 0.0]
	p=[β, 0.25]
	tspan = (0.0, float(l))
	prob = ODEProblem(sir_ode!, u0, tspan, p)
	sol = solve(prob, Tsit5(), saveat = 1.0)
	sol_C = Array(sol)[4,:] # Cumulative cases
	sol_X = sol_C[2:end] - sol_C[1:(end-1)]
	
	l = length(y)
	if any(sol_X .< 0)
		# Check if we have negative cumulative cases
		Turing.acclogp!(_varinfo, -Inf)
		return l
	end
	
	phi = 0.5
	variance = sol_X .+ sol_X.^2 ./ phi
	rs = neg_bin_r.(sol_X, variance)
	ps = neg_bin_p.(rs, sol_X)
	if !all(rs .> 0)
		Turing.acclogp!(_varinfo, -Inf)
		@warn "This shouldn't happen" β i₀
		@show sol_X
		return l
	end
	
	y ~ arraydist(NegativeBinomial.(rs, ps))
	#y ~ arraydist([Poisson(x) for x in sol_X])
end

# ╔═╡ a2be3cbe-f50d-11ea-2d63-2377fd79d4ce
begin
	ode_prior = sample(bayes_sir(Y, total_population), Prior(), 1000)
	plot_predictive(obstimes, X, Y, ode_prior)
end

# ╔═╡ 1de947a6-f5d5-11ea-26f5-d9066e89bbe7
begin
	R0s = base_reproduction_rate.(Array(ode_prior[:β]), p[2])
	histogram(R0s, bins=50)
end

# ╔═╡ 265cbe54-f508-11ea-2d8e-8f05cf5ec2b7
cost_fn(x; k=1) = 1_000_000 * logistic(k*(10*base_reproduction_rate(x, p[2]) - 13))

# ╔═╡ 2520b6c8-f504-11ea-2789-cf85d9f43b11
function marginal(beta, logjoint)
	lj_evals = Float64[]
    function marg(x, p)
        lj = logjoint([x[1], beta])
        push!(lj_evals, lj)
        return lj
    end
	
	marg_sol = solve(
		QuadratureProblem(marg, zeros(1), 1_000*ones(1)),
		HCubatureJL(), reltol=1e-5, abstol=1e-5
	)
	return logsumexp(lj_evals)
end

# ╔═╡ 650c886e-f5ba-11ea-1afc-47a86cce8de5
begin
	xs2 = collect(range(0.1,0.5,length=100))
	plot(
		xs2, 
		cost_fn.(xs2), 
		lw=3, 
		title="cost(beta)",
		legend=false
	)
end

# ╔═╡ 95acbe60-f36e-11ea-1126-0b6a5491c046
spl = Turing.Sampler(NUTS(0.65), bayes_sir(Y, total_population));

# ╔═╡ d26c56d4-f36f-11ea-3f35-ad9501efa7d8
log_density = gen_logjoint(spl.state.vi, bayes_sir(Y, total_population), spl);

# ╔═╡ 05b1fa20-f36c-11ea-0885-57b6426ad385
begin
	i0s = range(2, 20, length=100)
	betas = range(0, 1, length=100)
	grid2d = Iterators.product(
		i0s,
		betas
	)
	#grid2d = map(x -> (i₀=x[1], β=x[2]), grid2d)
	joint_vals = map(log_density, grid2d)
	heatmap(
		betas, 
		i0s, 
		joint_vals,
		ylabel="i0",
		xlabel="beta",
		title="logjoint"
	)
	scatter!([beta], [i0], marker=:x, legend=false, color=:black)
end

# ╔═╡ 3e6d6cc6-f597-11ea-2c9b-a5c1def2a4d7
begin
	jointf_vals = map(x -> log_density(x) + log(cost_fn(x[2])), grid2d)
	heatmap(
		betas, 
		i0s, 
		jointf_vals,
		ylabel="i0",
		xlabel="beta",
		title="joint + cost_fn"
	)
end

# ╔═╡ 8a6cad86-f505-11ea-3b33-c138ff125e28
function exp1_target(beta)
	return marginal(beta, log_density) + log(cost_fn(beta))
end

# ╔═╡ b2f523b6-f505-11ea-0704-97abd22517ee
function exp2_target(beta)
	return marginal(beta, log_density)
end

# ╔═╡ caf56bec-f505-11ea-330b-ef04fe4b3b44
begin
	xs = collect(range(0.028, 0.035, length=100))
	pl2 = plot(
		xs, 
		exp1_target.(xs), 
		lw=3, 
		label="p(beta,Y)f(beta)",
		ylims=[0,maximum(exp1_target.(xs))]
	)
	plot!(pl2, xs, exp2_target.(xs), lw=3, label="p(beta,Y)")
end

# ╔═╡ d14bd810-f5ba-11ea-24c8-878979227695
begin
	plot(
		xs2, 
		exp2_target.(xs2), 
		lw=3, 
		title="log(p(beta,Y))",
		legend=false
	)
end

# ╔═╡ 3934b9fe-f5bb-11ea-0187-0554bf7c2463
begin
	plot(
		xs2, 
		exp1_target.(xs2), 
		lw=3, 
		title="log(p(beta,Y)) + log(cost(beta))",
		legend=false
	)
end

# ╔═╡ 42fe2a66-f59c-11ea-2e00-07856abe20df
log_prior = gen_logprior(spl.state.vi, bayes_sir(Y, total_population), spl);

# ╔═╡ 35f4845a-f59c-11ea-0dcc-c55a651e2841
begin
	prior_vals = map(log_prior, grid2d)
	heatmap(
		betas,
		i0s, 
		prior_vals,
		ylabel="i0",
		xlabel="beta",
		title="logprior"
	)
end

# ╔═╡ abcfd7d2-f372-11ea-1163-21cb7a2064d2
est_exp(post_samples, f) = mean(f.(post_samples))

# ╔═╡ ab3dc0fe-f4da-11ea-1898-0f18798db60e
function numeric_expectation_calculation(logjoint, f)
	i_max = 1_000
	beta_max = 20
	params_max = [i_max, beta_max]
	
    norm_const_evals = Float64[]
    function norm_const_fn(x, p)
        lj = logjoint(x)
        push!(norm_const_evals, lj)
        return exp(lj)
    end
    
    norm_solve = solve(
		QuadratureProblem(norm_const_fn, zeros(2), params_max),
		HCubatureJL(), reltol=1e-5, abstol=1e-5
	)
    Z = norm_solve.u
    Z_logsumexp = logsumexp(norm_const_evals)

    unnorm_exp_evals = Float64[]
    function unnorm_exp_fn(x, p)
        val = logjoint(x) + log(f(x))
        push!(unnorm_exp_evals, val)
        return exp(val)
    end
    unnorm_exp_solve = solve(
		QuadratureProblem(
            unnorm_exp_fn, 
            zeros(2), 
            params_max
        ),
		HCubatureJL(), reltol=1e-5, abstol=1e-5
	)
    exp_gamma_f = unnorm_exp_solve.u
    exp_gamma_f_logsumexp = logsumexp(unnorm_exp_evals)
    
    return exp_gamma_f / Z, exp(exp_gamma_f_logsumexp - Z_logsumexp), exp(Z_logsumexp)
end

# ╔═╡ fd7576cc-f805-11ea-30bd-495dad3989fa
function numeric_expectation_calculation_marginal(logjoint, f)
	beta_max = 20
	params_max = [beta_max]
	
    norm_const_evals = Float64[]
    function norm_const_fn(x, p)
        lj = logjoint(x[1])
        push!(norm_const_evals, lj)
        return exp(lj)
    end
    
    norm_solve = solve(
		QuadratureProblem(norm_const_fn, zeros(1), params_max),
		HCubatureJL(), reltol=1e-5, abstol=1e-5
	)
    Z = norm_solve.u
    Z_logsumexp = logsumexp(norm_const_evals)

    unnorm_exp_evals = Float64[]
    function unnorm_exp_fn(x, p)
        val = logjoint(x[1]) + log(f(x[1]))
        push!(unnorm_exp_evals, val)
        return exp(val)
    end
    unnorm_exp_solve = solve(
		QuadratureProblem(
            unnorm_exp_fn, 
            zeros(1), 
            params_max
        ),
		HCubatureJL(), reltol=1e-5, abstol=1e-5
	)
    exp_gamma_f = unnorm_exp_solve.u
    exp_gamma_f_logsumexp = logsumexp(unnorm_exp_evals)
    
    return exp_gamma_f / Z, exp(exp_gamma_f_logsumexp - Z_logsumexp), exp(Z_logsumexp)
end

# ╔═╡ 3c823daa-f4dc-11ea-1e4a-fdf09583c38f
md"Numeric calculation of expectation:"

# ╔═╡ 20040204-f4db-11ea-0080-cdedd1bd857a
numeric_expectation_calculation(log_density, x -> cost_fn(x[2]))[2]

# ╔═╡ 4083a0da-f806-11ea-213e-15ed3f48d238
numeric_expectation_calculation_marginal(x -> marginal(x, log_density), cost_fn)[2]

# ╔═╡ 12a68ccc-f7f0-11ea-0eae-5f53dc0ef9d1
log_Z = log(numeric_expectation_calculation_marginal(x -> marginal(x, log_density), cost_fn)[3])

# ╔═╡ 08389d2a-f7f0-11ea-367f-1fb003a93e90
begin
	plot(
		xs2, 
		exp.(exp2_target.(xs2) .- log_Z), 
		lw=3, 
		title="p(beta,Y)",
		legend=false
	)
end

# ╔═╡ a6cf0b40-f7f0-11ea-2474-711420cf4b9e
begin
	plot(
		xs2, 
		exp.(exp1_target.(xs2) .- log_Z), 
		lw=3, 
		title="p(beta,Y) * cost(beta)",
		legend=false
	)
end

# ╔═╡ 87a596d8-f598-11ea-1955-317bbad35c98
md"Playing with prior distributions"

# ╔═╡ c99dbdc4-f5b1-11ea-0322-eb9bb8662800
@bind observed_X Slider(1:400)

# ╔═╡ 59f1f87c-f5b2-11ea-0a4b-1de1abbce8fd
observed_X

# ╔═╡ 1a82547a-f5b2-11ea-0cae-576dcb70a237
@bind phi_inv Slider(0:0.01:2)

# ╔═╡ 474f42b8-f5b2-11ea-2644-81d2024027bf
phi_inv

# ╔═╡ 064b78e8-f5ac-11ea-2381-bbc85adf1820
X_lik = NegativeBinomial(neg_bin_params2(observed_X, 1 / phi_inv)...)
#X_lik = Poisson(observed_X)

# ╔═╡ 47892436-f5ac-11ea-1c83-8fcea42a52ee
begin
	Xs = collect(1:400)
	bar(Xs, pdf.(X_lik, Xs))
end

# ╔═╡ Cell order:
# ╠═70d9471e-f361-11ea-3d79-857e54fd86ec
# ╟─87b490e0-f363-11ea-3598-f772c67da114
# ╟─bf72c778-f362-11ea-1763-f1c7c2b761fb
# ╟─851de356-f366-11ea-3bfa-6f28b87378bf
# ╟─8acada86-f50d-11ea-2979-2f821a08aee0
# ╟─97441d98-f50d-11ea-364b-0b9955b8afff
# ╟─43713eae-f36f-11ea-07e5-3143c8a802cd
# ╟─4f3a5b1a-f59c-11ea-2121-4bff69f69ddc
# ╠═beb376b6-f362-11ea-3df6-611ae46edd16
# ╠═40c3a9c8-f363-11ea-3056-0172f86c11f3
# ╠═5020c414-f363-11ea-3d8d-3d124c684663
# ╠═aa483766-f367-11ea-21e8-a3094913df8c
# ╠═b6505484-f363-11ea-39c2-cf47a29fd04c
# ╟─98532414-f365-11ea-05e7-f3adfa263d75
# ╠═4fae7a58-f363-11ea-2f53-e3e7406a27e4
# ╠═09830f94-f368-11ea-0356-d7bb1bec4702
# ╟─f9443cd0-f366-11ea-11db-152aeab61686
# ╟─4f745940-f363-11ea-136d-cb3d8ef9dec2
# ╠═a731c82c-f366-11ea-06c1-d77b42a7ad86
# ╠═5d23f438-f363-11ea-03b6-01c3c4a5b784
# ╠═22f63156-f366-11ea-33d6-193ab9db59a9
# ╠═7883dd38-f363-11ea-2dba-219d4ecc794d
# ╠═7a9b89fe-f363-11ea-370a-75afb531a410
# ╠═8cc14074-f363-11ea-2523-13016e95943b
# ╠═666ee530-f50b-11ea-00aa-eb40e3f3cdd3
# ╠═920a63c6-f363-11ea-3c99-a97f3db6d4c0
# ╟─302e99b0-f366-11ea-3110-43325fee9342
# ╟─07fc5356-f50a-11ea-0ef1-2dbf0cb60038
# ╟─2ae8e6f6-f50a-11ea-02b5-19ddd496b7aa
# ╟─52d0d8ae-f50a-11ea-0939-d95feb97d248
# ╟─e24e1b1a-f5d0-11ea-10dc-b5e3ba2652ae
# ╠═d3a7154c-f36b-11ea-21f6-1569109c69eb
# ╠═a2be3cbe-f50d-11ea-2d63-2377fd79d4ce
# ╠═1de947a6-f5d5-11ea-26f5-d9066e89bbe7
# ╟─35f4845a-f59c-11ea-0dcc-c55a651e2841
# ╟─05b1fa20-f36c-11ea-0885-57b6426ad385
# ╟─3e6d6cc6-f597-11ea-2c9b-a5c1def2a4d7
# ╠═265cbe54-f508-11ea-2d8e-8f05cf5ec2b7
# ╟─2520b6c8-f504-11ea-2789-cf85d9f43b11
# ╟─8a6cad86-f505-11ea-3b33-c138ff125e28
# ╟─b2f523b6-f505-11ea-0704-97abd22517ee
# ╟─caf56bec-f505-11ea-330b-ef04fe4b3b44
# ╟─650c886e-f5ba-11ea-1afc-47a86cce8de5
# ╟─d14bd810-f5ba-11ea-24c8-878979227695
# ╟─3934b9fe-f5bb-11ea-0187-0554bf7c2463
# ╠═95acbe60-f36e-11ea-1126-0b6a5491c046
# ╠═d26c56d4-f36f-11ea-3f35-ad9501efa7d8
# ╠═42fe2a66-f59c-11ea-2e00-07856abe20df
# ╟─abcfd7d2-f372-11ea-1163-21cb7a2064d2
# ╠═ab3dc0fe-f4da-11ea-1898-0f18798db60e
# ╠═fd7576cc-f805-11ea-30bd-495dad3989fa
# ╟─3c823daa-f4dc-11ea-1e4a-fdf09583c38f
# ╠═20040204-f4db-11ea-0080-cdedd1bd857a
# ╠═4083a0da-f806-11ea-213e-15ed3f48d238
# ╠═12a68ccc-f7f0-11ea-0eae-5f53dc0ef9d1
# ╟─08389d2a-f7f0-11ea-367f-1fb003a93e90
# ╟─a6cf0b40-f7f0-11ea-2474-711420cf4b9e
# ╟─87a596d8-f598-11ea-1955-317bbad35c98
# ╠═c99dbdc4-f5b1-11ea-0322-eb9bb8662800
# ╟─59f1f87c-f5b2-11ea-0a4b-1de1abbce8fd
# ╠═1a82547a-f5b2-11ea-0cae-576dcb70a237
# ╠═474f42b8-f5b2-11ea-2644-81d2024027bf
# ╠═064b78e8-f5ac-11ea-2381-bbc85adf1820
# ╠═47892436-f5ac-11ea-1c83-8fcea42a52ee
