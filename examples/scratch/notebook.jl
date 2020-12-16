### A Pluto.jl notebook ###
# v0.11.3

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

# ╔═╡ 1ea102a0-e6de-11ea-32db-35a606bb35f3
let
	import Pkg
	Pkg.activate(".")
	using Plots
	using Distributions
	using PlutoUI
	using Random
end

# ╔═╡ 4e863360-d7b2-11ea-2313-89b5dcba3c73
let
	cd("/Users/reichelt/dev/projects/julia/ExpectationProgramming/examples/scratch")
	import Pkg
	Pkg.activate(".")
	using Plots
	using Distributions
	using PlutoUI
	using ExpectationProgramming
	using AnnealedIS
	using Random
end

# ╔═╡ 3301d328-e6de-11ea-20b4-ddef56a66476
@bind mu Slider(-5:5)

# ╔═╡ 66aadf12-e6de-11ea-3f2e-6518928a2c78
md"mu: $(mu)"

# ╔═╡ 4be8789c-e6de-11ea-0690-1f6131e878b9
@bind sigma Slider(0:0.25:5)

# ╔═╡ 7a52bb0c-e6de-11ea-2f50-15ab817f22ce
md"sigma: $(sigma)"

# ╔═╡ 8b267056-e6de-11ea-1cd4-138b58a9382f
x_points = 0:0.1:5

# ╔═╡ ae5fa022-e6de-11ea-0464-05f87769dc25
ln_dist = LogNormal(mu, sigma)

# ╔═╡ a902ece4-e6de-11ea-203d-71310b7cd7e2
plot(x_points, pdf.(ln_dist, x_points))

# ╔═╡ d4a6eea6-d7da-11ea-169c-154b4ced8294
Random.seed!(42)

# ╔═╡ 3d6676c8-d7bd-11ea-3bbd-fded8345726e
Plots.default(lw=2)

# ╔═╡ d5643686-d7b3-11ea-2734-f556b544ea31
prior = Normal(0, 1)

# ╔═╡ f25758c2-d7b3-11ea-3ef9-a74a5d6046c6
likelihood(x) = Normal(x, 1)

# ╔═╡ 178a568a-d7b4-11ea-2fd7-69e135b0ffc2
@bind y_obs Slider(0:10)

# ╔═╡ 7b0a364a-d7b8-11ea-06e2-b993d485fcad
md"Observe at $(y_obs)"

# ╔═╡ 035f282a-d7b4-11ea-0a57-7fc4d40e7784
joint(x) = pdf(prior, x) * pdf(likelihood(x), y_obs)

# ╔═╡ b59ed37c-d7b5-11ea-0d4b-c5e2114ddd58
posterior = Normal(0.5*y_obs, sqrt(1.5))

# ╔═╡ 2fcbac9e-d7b4-11ea-1ee4-739cee9c252b
xs = -4:0.2:5

# ╔═╡ 5751238c-d7b4-11ea-20a9-8574944830a0
begin
	plot(xs, pdf.(prior, xs), label="p(x)")
	plot!(xs, pdf.(likelihood.(xs),y_obs), label="p(y|x)")
	plot!(xs, joint.(xs), label="p(x,y)")
	plot!(xs, pdf.(posterior, xs), label="p(x|y)")
end

# ╔═╡ ad50888e-d7b8-11ea-2685-6de03886c2df
prior_samples = rand(prior, 100)

# ╔═╡ ab9a4e98-d7b9-11ea-0241-1b5abe54b8cb
accepted_samples = filter(x -> x > 0, prior_samples)

# ╔═╡ b9fd5926-d7b9-11ea-129e-c9f6234c159a
rejected_samples = filter(x -> x <= 0, prior_samples)

# ╔═╡ 1d144b14-d7bc-11ea-3f85-cd30ae5846d5
joint_f_scale_factor = 13

# ╔═╡ c2e1f89c-d7b8-11ea-0ae2-87872b44be41
begin
	plot(xs, pdf.(prior, xs), label="p(x)")
	plot!(xs, max.(xs, 0) .* pdf.(prior, xs), label="p(x)f(x)")
	plot!(xs, joint_f_scale_factor * joint.(xs) .* max.(xs,0), label="p(x,y)f(x)*$(joint_f_scale_factor)")
	scatter!(accepted_samples, zeros(length(accepted_samples)), label="Accepted samples", alpha = 0.6)
	scatter!(rejected_samples, zeros(length(rejected_samples)), label="Rejected samples", alpha = 0.6)
end

# ╔═╡ 3afbcfe4-d7dd-11ea-34c3-678e45b81177
@expectation function expct(y)
    x ~ Normal(0, 1) 
    y ~ Normal(x, 1)
    return x
end

# ╔═╡ 5087a540-d7dd-11ea-2e6c-9544ad482592
expct_conditioned = expct(3)

# ╔═╡ 5ee23d3a-d7dd-11ea-0182-1561ae6d21f6
num_annealing_dists = 100

# ╔═╡ 6c487bb0-d7dd-11ea-0f13-95f41d7e61b6
num_samples = 1000

# ╔═╡ 7fce6faa-d7dd-11ea-3cfb-9be12f7d95c9
ais = AnnealedISSampler(expct_conditioned.gamma2, num_annealing_dists)

# ╔═╡ 824ebc08-d7dd-11ea-0bd3-b78d5599f3da
samples, diagnostics = ais_sample(
	Random.GLOBAL_RNG, ais, num_samples; store_intermediate_samples=true)

# ╔═╡ 2d1feac2-d7df-11ea-203e-1b4c7c379675
inter_samples = permutedims(hcat(diagnostics[:intermediate_samples]...), [2,1])

# ╔═╡ 13b27516-d7e0-11ea-05a1-b5f196e128dc
size(inter_samples)

# ╔═╡ f3fb445a-d7df-11ea-0c8d-8d5f94f70919
begin
	p = plot(title="Intermediate Samples", legend=false)
	for i in 1:size(inter_samples, 2)
		scatter!(
			p, 
			map(x->x[:x], inter_samples[:,i]), 
			ones(num_samples)*i, 
			alpha=0.2,
			color=:black
		)
	end
	p
end

# ╔═╡ 4b3e732a-d878-11ea-0b9f-e15ab06eae59
tabi = TABI(
	AIS(num_samples, num_annealing_dists)
)

# ╔═╡ 67c53b44-d878-11ea-2a0e-fb8f30324338
Z_tabi_estimate, tabi_diagnostics = estimate(expct_conditioned, tabi; store_intermediate_samples=true)

# ╔═╡ 7663e7fe-d87a-11ea-2518-6bce34f96c0b
Z1_positive_inter_samples = permutedims(
	hcat(tabi_diagnostics[:Z1_positive_info][:intermediate_samples]...), 
	[2,1]
)

# ╔═╡ b40d4df2-d87a-11ea-0942-e34656171134
Z1_samples = tabi_diagnostics[:Z1_positive_info][:samples]

# ╔═╡ 8ff85828-d87b-11ea-26ba-e5c9d131bed6
rejected_ixs = filter(ix -> Z1_samples[ix].log_weight == -Inf, 1:num_samples)

# ╔═╡ fc46ac0c-d87b-11ea-0d2a-9b0ddc2bea1a
accepted_ixs = setdiff(1:num_samples, rejected_ixs)

# ╔═╡ 199989a0-d87c-11ea-2048-7dc466dcca05
begin
	p2 = plot(legend=false)
	ylabel!(p2, "Distribution Index")
	scatter!(
		p2, 
		map(x->x[:x], Z1_positive_inter_samples[rejected_ixs,1]), 
		zeros(num_samples), 
		alpha=0.2,
		color=:red
	)
	for i in 1:size(Z1_positive_inter_samples, 2)
		scatter!(
			p2, 
			map(x->x[:x], Z1_positive_inter_samples[accepted_ixs,i]), 
			ones(num_samples)*(i-1)*10, 
			alpha=0.2,
			color=:black
		)
	end
	p2
end

# ╔═╡ fde16262-d87d-11ea-37e1-691db9c38ea6
begin
	p_Z1d = plot(xs, pdf.(prior, xs), label="p(x)")
	xlabel!(p_Z1d, "x")
	plot!(
		p_Z1d, 
		xs, 
		joint_f_scale_factor * joint.(xs) .* max.(xs,0), 
		label="p(x,y)f+(x)*$(joint_f_scale_factor)"
	)
	plot(
		p2, 
		p_Z1d, 
		layout=grid(2, 1, heights=[0.7, 0.3]), 
		link=:x
	)
	savefig("Z1_positive_samples.pdf")
end

# ╔═╡ c136d1b6-d881-11ea-3e7d-bd93035f108e
Z2_inter_samples = permutedims(
	hcat(tabi_diagnostics[:Z2_info][:intermediate_samples]...), 
	[2,1]
)

# ╔═╡ fd42f8d6-d87d-11ea-16a2-a7d453fa3e07
Z2_samples = tabi_diagnostics[:Z2_info][:samples]

# ╔═╡ 708c2c2c-d881-11ea-078e-613d642d3468
Z2_rejected_ixs = filter(ix -> Z2_samples[ix].log_weight == -Inf, 1:num_samples)

# ╔═╡ 8b1d30d4-d881-11ea-2378-97fed763a284
Z2_accepted_ixs = setdiff(1:num_samples, Z2_rejected_ixs)

# ╔═╡ 988b55f4-d881-11ea-11fe-31ce560e9c20
begin
	p_Z2 = plot(legend=false)
	ylabel!(p_Z2, "Distribution Index")
	for i in 1:size(Z2_inter_samples, 2)
		scatter!(
			p_Z2, 
			map(x->x[:x], Z2_inter_samples[:,i]), 
			ones(num_samples)*(i-1)*10, 
			alpha=0.2,
			color=:black
		)
	end
	p_Z2d = plot(xs, pdf.(prior, xs), label="p(x)")
	xlabel!(p_Z2d, "x")
	plot!(
		p_Z2d, 
		xs, 
		joint_f_scale_factor * joint.(xs), 
		label="p(x,y)*$(joint_f_scale_factor)"
	)
	plot(
		p_Z2, 
		p_Z2d, 
		layout=grid(2, 1, heights=[0.7, 0.3]), 
		link=:x
	)
	savefig("Z2_samples.pdf")
end

# ╔═╡ c0b5bd48-d882-11ea-36a7-f166685b2215
Z1_negative_inter_samples = permutedims(
	hcat(tabi_diagnostics[:Z1_negative_info][:intermediate_samples]...), 
	[2,1]
)

# ╔═╡ d3b95f62-d882-11ea-2772-976fde3f60c6
Z1n_samples = tabi_diagnostics[:Z1_negative_info][:samples]

# ╔═╡ dcd4587c-d882-11ea-3cb0-5b9b266310d3
Z1n_rejected_ixs = filter(ix -> Z1n_samples[ix].log_weight == -Inf, 1:num_samples)

# ╔═╡ e7c0dfc8-d882-11ea-33a6-bf0a8714c648
Z1n_accepted_ixs = setdiff(1:num_samples, Z1n_rejected_ixs)

# ╔═╡ 7b5d6920-d883-11ea-0cf1-2f22a5b58b80
scale_factor = 1000

# ╔═╡ f801689a-d882-11ea-0d50-8d1b0c8725bb
begin
	p_Z1n = plot(legend=false)
	ylabel!(p_Z1n, "Distribution Index")
	scatter!(
		p_Z1n, 
		map(x->x[:x], Z1_negative_inter_samples[Z1n_rejected_ixs,1]), 
		zeros(num_samples), 
		alpha=0.2,
		color=:red
	)
	for i in 1:size(Z1_negative_inter_samples, 2)
		scatter!(
			p_Z1n, 
			map(x->x[:x], Z1_negative_inter_samples[Z1n_accepted_ixs,i]), 
			ones(num_samples)*(i-1)*10, 
			alpha=0.2,
			color=:black
		)
	end
	p_Z1nd = plot(xs, pdf.(prior, xs), label="p(x)")
	xlabel!(p_Z1nd, "x")
	plot!(
		p_Z1nd, 
		xs, 
		scale_factor * joint.(xs) .* -min.(xs,0), 
		label="p(x,y)*f-(x)*$(scale_factor)"
	)
	plot(
		p_Z1n, 
		p_Z1nd, 
		layout=grid(2, 1, heights=[0.7, 0.3]), 
		link=:x
	)
	savefig("Z1_negative_samples.pdf")
end

# ╔═╡ Cell order:
# ╠═1ea102a0-e6de-11ea-32db-35a606bb35f3
# ╟─3301d328-e6de-11ea-20b4-ddef56a66476
# ╟─66aadf12-e6de-11ea-3f2e-6518928a2c78
# ╟─4be8789c-e6de-11ea-0690-1f6131e878b9
# ╟─7a52bb0c-e6de-11ea-2f50-15ab817f22ce
# ╠═8b267056-e6de-11ea-1cd4-138b58a9382f
# ╠═ae5fa022-e6de-11ea-0464-05f87769dc25
# ╠═a902ece4-e6de-11ea-203d-71310b7cd7e2
# ╠═4e863360-d7b2-11ea-2313-89b5dcba3c73
# ╠═d4a6eea6-d7da-11ea-169c-154b4ced8294
# ╠═3d6676c8-d7bd-11ea-3bbd-fded8345726e
# ╠═d5643686-d7b3-11ea-2734-f556b544ea31
# ╠═f25758c2-d7b3-11ea-3ef9-a74a5d6046c6
# ╠═178a568a-d7b4-11ea-2fd7-69e135b0ffc2
# ╟─7b0a364a-d7b8-11ea-06e2-b993d485fcad
# ╠═035f282a-d7b4-11ea-0a57-7fc4d40e7784
# ╠═b59ed37c-d7b5-11ea-0d4b-c5e2114ddd58
# ╠═2fcbac9e-d7b4-11ea-1ee4-739cee9c252b
# ╠═5751238c-d7b4-11ea-20a9-8574944830a0
# ╠═ad50888e-d7b8-11ea-2685-6de03886c2df
# ╠═ab9a4e98-d7b9-11ea-0241-1b5abe54b8cb
# ╠═b9fd5926-d7b9-11ea-129e-c9f6234c159a
# ╠═1d144b14-d7bc-11ea-3f85-cd30ae5846d5
# ╠═c2e1f89c-d7b8-11ea-0ae2-87872b44be41
# ╠═3afbcfe4-d7dd-11ea-34c3-678e45b81177
# ╠═5087a540-d7dd-11ea-2e6c-9544ad482592
# ╠═5ee23d3a-d7dd-11ea-0182-1561ae6d21f6
# ╠═6c487bb0-d7dd-11ea-0f13-95f41d7e61b6
# ╠═7fce6faa-d7dd-11ea-3cfb-9be12f7d95c9
# ╠═824ebc08-d7dd-11ea-0bd3-b78d5599f3da
# ╠═2d1feac2-d7df-11ea-203e-1b4c7c379675
# ╠═13b27516-d7e0-11ea-05a1-b5f196e128dc
# ╠═f3fb445a-d7df-11ea-0c8d-8d5f94f70919
# ╠═4b3e732a-d878-11ea-0b9f-e15ab06eae59
# ╠═67c53b44-d878-11ea-2a0e-fb8f30324338
# ╠═7663e7fe-d87a-11ea-2518-6bce34f96c0b
# ╠═b40d4df2-d87a-11ea-0942-e34656171134
# ╠═8ff85828-d87b-11ea-26ba-e5c9d131bed6
# ╠═fc46ac0c-d87b-11ea-0d2a-9b0ddc2bea1a
# ╠═199989a0-d87c-11ea-2048-7dc466dcca05
# ╠═fde16262-d87d-11ea-37e1-691db9c38ea6
# ╠═c136d1b6-d881-11ea-3e7d-bd93035f108e
# ╠═fd42f8d6-d87d-11ea-16a2-a7d453fa3e07
# ╠═708c2c2c-d881-11ea-078e-613d642d3468
# ╠═8b1d30d4-d881-11ea-2378-97fed763a284
# ╠═988b55f4-d881-11ea-11fe-31ce560e9c20
# ╠═c0b5bd48-d882-11ea-36a7-f166685b2215
# ╠═d3b95f62-d882-11ea-2772-976fde3f60c6
# ╠═dcd4587c-d882-11ea-3cb0-5b9b266310d3
# ╠═e7c0dfc8-d882-11ea-33a6-bf0a8714c648
# ╠═7b5d6920-d883-11ea-0cf1-2f22a5b58b80
# ╠═f801689a-d882-11ea-0d50-8d1b0c8725bb
