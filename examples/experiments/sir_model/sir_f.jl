using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using ExpectationProgramming
using AnnealedIS
using DataFrames
using StatsPlots
using StatsFuns: logsumexp, logistic
using StatsBase
using LinearAlgebra: dot
using Logging
using LoggingExtras
using AdvancedMH
using AdvancedHMC
using Dates
using JLD2
using FileIO
using Quadrature

const RANDOM_SEED = 1234
Random.seed!(RANDOM_SEED)

const RESULTS_FOLDER = "results_cost_fn/"

include("utils.jl")

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

function base_reproduction_rate(β, γ)
    return β / γ
end

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

function transition_kernel(prior_sample::T, i) where {T<:Real}
    if (i > 100) && (i <= 700)
        return Normal(0, 0.01)
    elseif (i > 700) 
        return Normal(0, 0.001)
    else
        return Normal(0, 0.1)
    end
end

function transition_kernel(prior_sample::NamedTuple, i)
    #return map(x -> transition_kernel(x, i), prior_sample)
    return (
        i₀ = Normal(0, 1),
        β = Normal(0, 0.1)
    )
end

function cost_fn(β, γ; k=1) 
    return 1_000_000 * logistic(k*(10*base_reproduction_rate(β, γ) - 10))
end

cost_fn2(β, γ; k=1) = 1_000_000_000_000 * logistic(k*(10*base_reproduction_rate(β, γ) - 30))

neg_bin_r(mean, var) = mean^2 / (var - mean)
neg_bin_p(r, mean) = r / (r + mean)

function neg_bin_params(mean, var)
    r = neg_bin_r(mean, var)
    return r, neg_bin_p(r, mean)
end

neg_bin_params2(mean, phi) = neg_bin_params(mean, mean + mean^2 / phi)

function is_sampling(logjoint, qi, qb, num_samples)
    qi_samples = rand(qi, num_samples)
    qb_samples = rand(qb, num_samples)

    is_weights = logjoint.(zip(qi_samples, qb_samples)) - (logpdf.(qi, qi_samples) .+ logpdf.(qb, qb_samples))
    normalised_is_weights = is_weights .- logsumexp(is_weights)
    
    return normalised_is_weights, qi_samples, qb_samples
end

function main(
    experiment_name; 
    plot_prior_pred=false, 
    sample_nuts=false,
    sample_mh=false,
    plot_joint_and_prior=false,
    sample_mh_intermediate=false,
    sample_is=false
)
    tmax = 15.0
    tspan = (0.0,tmax)
    obstimes = 1.0:1.0:tmax

    total_population = 10_000

    i_0_true = 10
    I0 = i_0_true * 10
	S0 = total_population - I0
	u0 = [S0, I0, 0.0, 0.0] # S,I.R,C

    β_true = 0.25
    # Fixed parameters.
    γ = 0.25
    p = [β_true, γ]

    prob_ode = ODEProblem(sir_ode!,u0,tspan,p)
    sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0)

    C = Array(sol_ode)[4,:] # Cumulative cases
    X = C[2:end] - C[1:(end-1)]
    #Y = rand.(Poisson.(X))
    Y = rand.([NegativeBinomial(neg_bin_params2(m, 10)...) for m in X])

    @expectation bayes_sir(y) = begin
        # Calculate number of timepoints
        l = length(y)
        i₀ ~ truncated(Normal(10, 10), 0, total_population/10)
        β ~ truncated(Normal(2, 1.5), 0, Inf)
        I = i₀ * 10
        u0=[total_population-I, I, 0.0, 0.0]
        p=[β, 0.25]
        tspan = (0.0, float(l))
        prob = ODEProblem(sir_ode!, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat = 1.0)
        sol_C = Array(sol)[4,:] # Cumulative cases
        sol_X = sol_C[2:end] - sol_C[1:(end-1)]
        
        l = length(y)
        if any(sol_X .< 0) || length(sol_X) != l 
            # Check if we have negative cumulative cases
            @warn "Negative cumulative cases" β i₀
            Turing.acclogp!(_varinfo, -Inf)
            return l
        end
        
        phi = 0.5
        variance = sol_X .+ sol_X.^2 ./ phi
        rs = neg_bin_r.(sol_X, variance)
        ps = neg_bin_p.(rs, sol_X)
        if !all(rs .> 0) || !all(0 .< ps .<= 1)
            Turing.acclogp!(_varinfo, -Inf)
            @warn "This shouldn't happen" β i₀
            return l
        end
        
        y ~ arraydist(NegativeBinomial.(rs, ps))
        #y ~ arraydist([Poisson(x) for x in sol_X])
        return cost_fn2(β, γ)
    end

    result_folder = make_experiment_folder(experiment_name)
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, "out.log"))
    )

    if plot_prior_pred
        ode_prior = sample(bayes_sir(Y).gamma2, Prior(), 1000)
        savefig(
            plot_predictive(obstimes, X, Y, ode_prior), 
            joinpath(result_folder, "prior_pred.png")
        )
        savefig(
            bar(
                obstimes, Y, xlabel="Day", ylabel="Number of cases", 
                legend=false,
                xtickfontsize=15,
                ytickfontsize=15,
                guidefontsize=15),
            joinpath(result_folder, "observed_data.pdf")
        )
    end

    if sample_nuts
        @time begin
        ode_nuts = mapreduce(c -> sample(
            bayes_sir(Y).gamma2, 
            Turing.NUTS(0.65), 
            1_000; 
            discard_adapt=true
        ), chainscat, 1:1)
        end
        savefig(
            plot_predictive(obstimes, X, Y, ode_nuts), 
            joinpath(result_folder, "nuts_post_pred.png")
        )
        savefig(
            plot(ode_nuts), 
            joinpath(result_folder, "nuts_traceplot.png")
        )
        write(joinpath(result_folder, "nuts_chain.jls"), ode_nuts)
        # Terminate the program early
        with_logger(logger) do
            @info "Chain info:" ode_nuts
        end
        return
    end

    if sample_is
        num_samples = Int(1e6)

        tm = bayes_sir(Y).gamma2
        spl = Turing.Sampler(Turing.NUTS(0.65), tm)
        joint_dens = AnnealedIS.gen_logjoint(spl.state.vi, tm, spl)

        qi = truncated(Normal(10, 10), 0, total_population/10)
        qb = truncated(Normal(1.5, 1.5), 0, Inf)

        @time begin
        log_weights, qi_samples, qb_samples = is_sampling(
            joint_dens, qi, qb, num_samples
        )
        end

        costs = cost_fn2.(qb_samples, γ)

        ess = 1 / exp(logsumexp(2 * log_weights))
        target_weights = log_weights .+ log.(costs)
        ess_target = exp(2 * logsumexp(target_weights) - logsumexp(2 * target_weights))

        exp_estimate = dot(exp.(log_weights), costs)
        with_logger(logger) do 
            @info "Proposals:" qi qb
            @info "Number of samples: $num_samples"
            @info "ESS p(x|y): $ess"
            @info "ESS p(x|y)f(x): $ess_target"
            @info "Expectation estimate: $exp_estimate"
        end

        save(
            joinpath(result_folder, "results.jld2"),
            "cost_fn", cost_fn2,
            "log_weights", log_weights,
            "qi_samples", qi_samples,
            "qb_samples", qb_samples
        )
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
        betas_intermediates = (geomspace(1, 1001, 400) .- 1) ./ 1_000
        beta1 = betas_intermediates[2]
        first_intermediate_density(params) = beta1 * joint_dens(params) + (1-beta1) * prior_dens(params)

        i0s = range(0, 1, length=100)
        betas = range(0, 1, length=100)
        grid2d = Iterators.product(
            i0s,
            betas
        )
        grid2d = map(x -> (i₀=x[1], β=x[2]), grid2d)
        prior_vals = map(prior_dens, grid2d)
        joint_vals = map(joint_dens, grid2d)
        first_vals = map(first_intermediate_density, grid2d)
        p = heatmap(
            i0s, 
            betas, 
            prior_vals,
            xlabel="beta",
            ylabel="i0"
        )
        savefig(
            p,
            joinpath(result_folder, "prior_density_heatmap.png")
        )
        p = heatmap(
            i0s, 
            betas, 
            joint_vals, 
            size=(1200,800),
            xlabel="beta",
            ylabel="i0"
        )
        savefig(
            p,
            joinpath(result_folder, "joint_density_heatmap.png")
        )
        p = heatmap(
            i0s, 
            betas, 
            first_vals, 
            size=(1200,800),
            xlabel="beta",
            ylabel="i0"
        )
        savefig(
            p,
            joinpath(result_folder, "first_intermediate_heatmap.png")
        )
        return
    end

    # Calculated with 1_000_000 NUTS samples.
    true_expectation_value = nothing
    
    num_samples = 1_000
    #num_annealing_dists = 200
    #anis_alg = AnIS(transition_kernel, num_annealing_dists, SimpleRejection())
    #betas = anis_alg.betas
    #betas_begin = collect(range(0, 0.01, length=40))
    #betas_end = collect(range(0.01, 1, length=162))
    #betas_end = (geomspace(1.01, 1001, 162) .- 1) ./ 1000
    #betas = vcat(betas_begin, betas_end[2:end])
    # We first create a geometric spacing between 1 and 1001 because directly 
    # doing it between 0 and 1 gives numerical problems.
    betas = (geomspace(1, 1001, 101) .- 1) ./ 1000
    #betas_begin = (geomspace(1, 1001, 990) .- 1) ./ 10_000
    #betas_end = collect(range(0.1, 1, length=12))
    #betas = vcat(betas_begin, betas_end[2:end])
    anis_alg = AnIS(betas, transition_kernel, SimpleRejection())
    #anis_alg = IS()

    # AnIS with HMC
    #proposal = AdvancedHMC.StaticTrajectory(AdvancedHMC.Leapfrog(0.005), 5)
    #proposal = AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(
    #    AdvancedHMC.Leapfrog(0.1)
    #)
    #anis_alg = AnISHMC(
    #    betas,
    #    proposal,
    #    10,
    #    SimpleRejection()
    #)

    tabi = TABI(
        TuringAlgorithm(anis_alg, num_samples),
        TuringAlgorithm(anis_alg, 0),
        TuringAlgorithm(anis_alg, num_samples)
    )

    @time begin
    R0_estimate, ode_tabi = estimate(
        bayes_sir(Y), tabi; store_intermediate_samples=true
    )
    end

    # Save experimental data
    save(
        joinpath(result_folder, "results.jld2"),
        "ode_tabi", ode_tabi,
        "R0_estimate", R0_estimate,
        "tabi", tabi,
        "obstimes", obstimes,
        "X", X,
        "Y", Y,
        "β_true", β_true,
        "i_0_true", i_0_true,
        "true_expectation_value", true_expectation_value
    )

    process_results(
        result_folder,
        ode_tabi=ode_tabi,
        R0_estimate=R0_estimate,
        true_expectation_value=true_expectation_value,
        tabi=tabi,
        X=X,
        Y=Y,
        obstimes=obstimes,
        i_0_true=i_0_true,
        β_true=β_true,
        cost_fn=((i, b) -> cost_fn2(b, γ))
    )
end

main(
    "$(RANDOM_SEED)_reproduce_anis_1000_samples";
    sample_nuts=false, 
    plot_prior_pred=true, 
    sample_is=false,
    sample_mh=false,
    plot_joint_and_prior=false,
    sample_mh_intermediate=false
)
