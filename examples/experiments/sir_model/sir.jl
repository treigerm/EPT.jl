# Code adapted from https://github.com/epirecipes/sir-julia/blob/master/markdown/ode_turing/ode_turing.md

using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using ExpectationProgramming
using AnnealedIS
using DataFrames
using StatsPlots
using StatsFuns: logsumexp
using Logging
using LoggingExtras

Random.seed!(1234)

const RESULTS_FOLDER = "results/"

function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end

function base_reproduction_rate(β, c, γ)
    return c * β / γ
end

function predict(y,chain)
    # Length of data
    l = length(y)
    # Length of chain
    m = length(chain)
    # Choose random
    idx = sample(1:m)
    i₀ = chain[:i₀].value[idx]
    β = chain[:β].value[idx]
    I = i₀*1000.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,10.0,0.25]
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

function plot_predictive(obstimes, X_true, Y_obs, chain)
    Xp = []
    for i in 1:10
        pred = predict(Y_obs, chain)
        push!(Xp, pred[2:end,6])
    end

    p = plot(obstimes, Xp; legend=false, color=:red, alpha=0.8)
    plot!(p, obstimes, X_true, color=:black, lw=3)
    scatter!(p, obstimes, Y_obs)
    return p
end

function plot_intermediate_samples(
    intermediate_samples, 
    param_name, 
    true_param,
    betas
)
    inter_samples = permutedims(hcat(intermediate_samples...), [2,1])

    num_samples = size(inter_samples, 1)
    p = plot(title="Intermediate samples for $(string(param_name))", legend=false)
    plot!(p, [true_param], color=:red, seriestype="vline")
    for i in 1:size(inter_samples, 2)
        ix = i > 10 ? (i-9)*10 : i
        beta = betas[ix]
		scatter!(
			p, 
			map(x->x[param_name], inter_samples[:,i]), 
			ones(num_samples)*beta, 
			alpha=0.2,
            color=:black,
            xlims=[-0.1,0.1],
            ylims=[0,1.0]
		)
    end
    return p
end

function plot_intermediate_weights(intermediate_weights, betas; ixs=nothing)
    inter_weights = permutedims(hcat(intermediate_weights...), [2,1])

    num_samples = size(inter_weights, 1)
    p = plot(
        title="Intermediate weights", 
        xlabel="betas",
        ylabel="log weight",
        legend=false
    )

    if isnothing(ixs) 
        ixs = 1:size(inter_weights, 2)
    end
	for i in ixs
        ix = i > 10 ? (i-9)*10 : i
        beta = betas[ix]
		scatter!(
			p,
			ones(num_samples)*beta, 
			inter_weights[:,i], 
			alpha=0.2,
            color=:black
		)
    end
    return p
end

function plot_intermediate_ess(intermediate_weights, betas; ixs=nothing)
    inter_weights = permutedims(hcat(intermediate_weights...), [2,1])

    ess = exp.(
        2 * logsumexp(inter_weights; dims=1) - logsumexp(2 * inter_weights; dims=1)
    )[1,:]

    if isnothing(ixs)
        ixs = 1:size(inter_weights, 2)
    end
    betas = [betas[i > 10 ? (i-9)*10 : i] for i in ixs]
    ess = ess[ixs]
    return plot(
        betas,
        ess,
        title="ESS for intermediate distributions",
        xlabel="beta",
        ylabel="ESS",
        lw=2,
        color=:black,
        legend=false
    )
end

function plot_joint_dist(chain, beta_true, i_0_true)
    betas = get(chain, :β)[:β]
    i_0s = get(chain, :i₀)[:i₀]
    p = scatter(
        betas, i_0s, xlabel="β", ylabel="i₀", label="AnIS samples", color=:black
    )
    scatter!(
        p, [beta_true], [i_0_true], label="True Params", color=:red, marker=:x
    )
    return p
end

function transition_kernel(prior_sample::T, i) where {T<:Real}
    if (i > 50) && (i <= 100)
        return Normal(0, 0.01)
    elseif (i > 100) 
        return Normal(0, 0.001)
    else
        return Normal(0, 0.1)
    end
end

function transition_kernel(prior_sample::NamedTuple, i)
    return map(x -> transition_kernel(x, i), prior_sample)
end

function make_experiment_folder(experiment_name)
    datestring = Dates.format(Dates.now(), "ddmmyyyy_HHMMSS")
    folder_name = "$(datestring)_$(experiment_name)"
    return mkpath(joinpath(RESULTS_FOLDER, folder_name))
end

function geomspace(start, stop, length)
    logstart = log10(start)
    logstop = log10(stop)
    points = 10 .^ range(logstart, logstop; length=length)
    points[1] = start
    points[end] = stop
    return points
end

function main(
    experiment_name; 
    plot_prior_pred=false, 
    sample_nuts=false,
    sample_mh=false,
    plot_joint_and_prior=false
)
    tmax = 40.0
    tspan = (0.0,tmax)
    obstimes = 1.0:1.0:tmax

    i_0_true = 0.01
    I_0 = 1000.0 * i_0_true
    S_0 = 1000.0 - I_0
    u0 = [S_0, I_0, 0.0, 0.0] # S,I.R,C
    β_true = 0.05
    p = [β_true, 10.0, 0.25]; # β,c,γ

    true_base_reproduction_rate = base_reproduction_rate(p...)

    prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
    sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0)

    C = Array(sol_ode)[4,:] # Cumulative cases
    X = C[2:end] - C[1:(end-1)]
    Y = rand.(Poisson.(X))

    #bar(obstimes,Y,legend=false)
    #plot!(obstimes,X,legend=false)

    @expectation function bayes_sir(y)
        # Calculate number of timepoints
        l = length(y)

        i₀ ~ Uniform(0.0,1.0)
        #β ~ Uniform(0.0,1.0)
        # Test to check whether more informative priors help with the very small 
        # weights.
        #i₀ ~ Beta(1, 3)
        β ~ Beta(2, 10)
        # True posterior marginals
        #i₀ ~ truncated(Normal(i_0_true, 0.01), 0, 1)
        #β ~ truncated(Normal(β_true, 0.01), 0, 1)


        if (i₀ > 1.0 || i₀ < 0.0) || (β > 1.0 || β < 0.0)
            Turing.acclogp!(_varinfo, -Inf)
            return l
        end

        # Fixed parameters.
        γ = 0.25
        c = 10.0

        I = i₀*1000.0
        S = 1000.0 - I
        u0 = [S, I, 0.0, 0.0]
        p = [β, c, γ]
        tspan = (0.0, float(l))

        prob = ODEProblem(sir_ode!, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat = 1.0)

        sol_C = Array(sol)[4,:] # Cumulative cases
        sol_X = sol_C[2:end] - sol_C[1:(end-1)]
        if any(sol_X .< 0)
            # Check if we have negative cumulative cases
            Turing.acclogp!(_varinfo, -Inf)
            return l
        end

        for i in 1:l
            y[i] ~ Poisson(sol_X[i])
        end

        return base_reproduction_rate(p...)
    end

    result_folder = make_experiment_folder(experiment_name)
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, "out.log"))
    )

    if plot_prior_pred
        ode_prior = sample(bayes_sir(Y).gamma2, Prior(), 100)
        savefig(
            plot_predictive(obstimes, X, Y, ode_prior), 
            joinpath(result_folder, "prior_pred.png")
        )
    end

    if sample_nuts
        ode_nuts = sample(bayes_sir(Y).gamma2, NUTS(0.65), 4000)
        savefig(
            plot_predictive(obstimes, X, Y, ode_nuts), 
            joinpath(result_folder, "nuts_post_pred.png")
        )
        savefig(
            ode_mh, 
            joinpath(result_folder, "nuts_traceplot.png")
        )
        write(joinpath(result_folder, "nuts_chain.jls"), ode_nuts)
        # Terminate the program early
        return
    end

    if sample_mh
        mh = MH(
            :i₀ => x -> Normal(x, 0.0001),
            :β => x -> Normal(x, 0.0001)
        )
        ode_mh = sample(bayes_sir(Y).gamma2, mh, 4000)
        savefig(
            plot_predictive(obstimes, X, Y, ode_mh),
            joinpath(result_folder, "mh_post_pred.png")
        )
        savefig(
            plot(ode_mh),
            joinpath(result_folder, "mh_traceplot.png")
        )
        with_logger(logger) do
            @info "Chain info:" ode_mh
        end
        return
    end

    if plot_joint_and_prior
        tm = bayes_sir(Y).gamma2
        prior_dens = AnnealedIS.make_log_prior_density(tm)
        joint_dens = AnnealedIS.make_log_joint_density(tm)
        i0s = range(0, 1, length=100)
        betas = range(0, 1, length=100)
        grid2d = Iterators.product(
            i0s,
            betas
        )
        grid2d = map(x -> (i₀=x[1], β=x[2]), grid2d)
        prior_vals = map(prior_dens, grid2d)
        joint_vals = map(joint_dens, grid2d)
        p = wireframe(i0s, betas, prior_vals)
        wireframe!(p, i0s, betas, joint_vals, color=:red)
        savefig(
            p,
            joinpath(result_folder, "prior_and_joint_density.png")
        )
        p = heatmap(i0s, betas, prior_vals)
        savefig(
            p,
            joinpath(result_folder, "prior_density_heatmap.png")
        )
        p = heatmap(i0s, betas, joint_vals)
        savefig(
            p,
            joinpath(result_folder, "joint_density_heatmap.png")
        )
        return
    end
    
    num_samples = 5
    #num_annealing_dists = 200
    #anis_alg = AnIS(transition_kernel, num_annealing_dists, SimpleRejection())
    #betas = anis_alg.betas
    #betas_begin = collect(range(0, 0.01, length=40))
    #betas_end = collect(range(0.01, 1, length=162))
    #betas_end = (geomspace(1.01, 1001, 162) .- 1) ./ 1000
    #betas = vcat(betas_begin, betas_end[2:end])
    # We first create a geometric spacing between 1 and 1001 because directly 
    # doing it between 0 and 1 gives numerical problems.
    betas = (geomspace(1, 1001, 401) .- 1) ./ 1000
    anis_alg = AnIS(betas, transition_kernel, SimpleRejection())
    #anis_alg = IS()
    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, 0),
        TuringAlgorithm(anis_alg, num_samples)
    )

    R0_estimate, ode_tabi = estimate(
        bayes_sir(Y), tabi; store_intermediate_samples=true
    )
    savefig(plot_intermediate_samples(
        ode_tabi[:Z2_info].info[:intermediate_samples], :β, β_true, betas),
        joinpath(result_folder, "intermediate_samples_beta.png")
    )
    savefig(plot_intermediate_samples(
        ode_tabi[:Z2_info].info[:intermediate_samples], :i₀, i_0_true, betas),
        joinpath(result_folder, "intermediate_samples_i_0.png")
    )
    savefig(plot_intermediate_weights(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "intermediate_weights.png")
    )
    savefig(plot_intermediate_weights(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas; ixs=1:14),
        joinpath(result_folder, "intermediate_weights_first40.png")
    )
    savefig(plot_intermediate_ess(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas),
        joinpath(result_folder, "intermediate_ess.png")
    )
    savefig(plot_intermediate_ess(
        ode_tabi[:Z2_info].info[:intermediate_log_weights], betas; ixs=1:14),
        joinpath(result_folder, "intermediate_ess_first40.png")
    )
    savefig(
        plot_predictive(obstimes, X, Y, ode_tabi[:Z2_info]), 
        joinpath(result_folder, "anis_post_pred.png")
    )
    savefig(
        plot_joint_dist(ode_tabi[:Z2_info], β_true, i_0_true),
        joinpath(result_folder, "joint_samples.png")
    )


    with_logger(logger) do
        @info "True expectation value: $(true_base_reproduction_rate)"
        @info "Expectation estimate: $(R0_estimate)"

        @info "Estimation algorithm: $(tabi)"

        for (k, est_results) in pairs(ode_tabi)
            if isnothing(est_results)
                continue
            end
            msg = string([
                "$(string(k)):\n", 
                "ESS: $(est_results.info[:ess])\n",
                "Log evidence: $(est_results.logevidence)\n",
                "Log weights: $(get(est_results, :log_weight)[:log_weight])\n"
            ]...)

            @info "$(msg)" 
        end
    end
end

main(
    "test_r0"; 
    sample_nuts=false, 
    plot_prior_pred=false, 
    sample_mh=false,
    plot_joint_and_prior=false
)