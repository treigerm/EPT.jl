using ExpectationProgramming
using AnnealedIS
using Distributions
using QuadGK
using Plots
using JLD

using Random
using Base.Threads
using Statistics: mean

#rng = MersenneTwister(42)
Random.seed!(42)

Plots.default(lw=2)

const NUM_ANNEALING_DISTS = 100
const NUM_SAMPLES = 10
const NUM_RUNS = 1

const RESULTS_FILE = "results.jld"

function compute_true_Zs(yval)
    Z1_plus_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval) * max(x,0)
    Z1_minus_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval) * (-min(x,0))
    Z2_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval)

    Z1_positive_true, _ = quadgk(Z1_plus_target, 0, 20, atol=1e-16)
    Z1_negative_true, _ = quadgk(Z1_minus_target, -10, 0, atol=1e-16)
    Z2_true, _ = quadgk(Z2_target, -10, 20, atol=1e-16)

    return (
        Z2_info = Z2_true,
        Z1_positive_info = Z1_positive_true,
        Z1_negative_info = Z1_negative_true
    )
end

function normalisation_constants(samples::Array{AnnealedIS.WeightedSample})
    weights = map(samples) do weighted_sample
        exp(weighted_sample.log_weight)
    end

    return cumsum(weights) ./ (1:length(samples))
end

function expectations(samples::Array{AnnealedIS.WeightedSample}, f)
    weighted_terms = map(samples) do weighted_sample
        exp(weighted_sample.log_weight) * f(weighted_sample.params)
    end

    weights = map(samples) do weighted_sample
        exp(weighted_sample.log_weight)
    end 

    return cumsum(weighted_terms) ./ cumsum(weights)
end

function convergence_plot(taanis_results, taanis_resample_results, ais_results)
    num_runs = length(taanis_results)
    num_samples_ais = length(ais_results[1][:Z][:samples])
    # TODO: Control number of samples when Z2 = 0.
    num_samples_taanis = Int(num_samples_ais / 3)
    taanis_intermediates = Array{Float64}(undef, num_runs, num_samples_taanis)
    taanis_resamples_intermediates = Array{Float64}(
        undef, num_runs, num_samples_taanis)
    ais_intermediates = Array{Float64}(undef, num_runs, num_samples_ais)

    for i in 1:num_runs
        ais_samples = ais_results[i][:Z][:samples]
        ais_intermediates[i,:] = expectations(ais_samples, x -> x[:x])

        Zs = (
            Z1_positive_info = Array{Float64}(undef, num_samples_taanis),
            Z1_negative_info = Array{Float64}(undef, num_samples_taanis),
            Z2_info = Array{Float64}(undef, num_samples_taanis)
        )
        for k in keys(Zs)
            samples = taanis_results[i][k][:samples]
            Zs[k][:] = normalisation_constants(samples)
        end
        taanis_intermediates[i,:] = (
            Zs[:Z1_positive_info] .- Zs[:Z1_negative_info]) ./ Zs[:Z2_info]
        
        for k in keys(Zs)
            samples = taanis_resample_results[i][k][:samples]
            Ns = 1:length(samples)
            rejections = taanis_resample_results[i][k][:num_rejected]
            acceptance_ratio = Ns ./ (Ns .+ cumsum(rejections))
            Zs[k][:] = acceptance_ratio .* normalisation_constants(samples)
        end
        taanis_resamples_intermediates[i,:] = (
            Zs[:Z1_positive_info] .- Zs[:Z1_negative_info]) ./ Zs[:Z2_info]
    end 

    # TODO: calculate error 
    # TODO: Make this plot more fancy
    p = plot(
        1:num_samples_ais, 
        ais_intermediates[1,:],
        label="AnIS"
    )
    plot!(
        p, 
        (1:num_samples_taanis) * 3, 
        taanis_resamples_intermediates[1,:],
        label="TAAnIS (resampling)"
    )
    plot!(
        p, 
        (1:num_samples_taanis) * 3, 
        taanis_intermediates[1,:],
        label="TAAnIS"
    )
    savefig(p, "convergence.png")
end

function relative_squared_error(x_hat, true_x) 
    return (x_hat - true_x)^2 / true_x^2
end

function display_results(
    results, # Dict with alg_name => expectation_estimates
    diagnostics, # Dict with alg_name => diagnostics
    true_Z, # True value for expectation
    true_Zs # True values of the individual normalisation constants for TABI
)
    for (name, v) in pairs(results)
        squared_errors = relative_squared_error.(v, true_Z)
        mean_error = mean(squared_errors)
        println("$name:")
        println("")
        println("Values: $v")
        println("Relative squared errors: $squared_errors")
        println("Mean relative squared errors: $mean_error")

        diags = diagnostics[name] # Array of NamedTuple
        println("ESS:")
        for estimator_name in keys(diags[1])
            ess_mean = mean(map(x -> x[estimator_name][:ess], diags))
            println("$estimator_name: $ess_mean")
        end

        if name in [:AIS, :AISResample]
            println("Error in individual estimates:")
            for estimator_name in keys(diags[1])
                errors = map(diags) do Z_est
                    relative_squared_error(
                        Z_est[estimator_name][:Z_estimate], 
                        true_Zs[estimator_name]
                    )
                end
                error_mean = mean(errors)
                
                println("$estimator_name: $error_mean")
            end
        end
        println("------------------------------")
    end
end

function main(num_annealing_dists, num_samples, num_runs)
    @expectation function expct(y)
        x ~ Normal(0, 1) 
        y ~ Normal(x, 1)
        return x
    end

    yval = 3
    expct_conditioned = expct(yval)

    true_x_posterior_mean = 1.5

    true_Zs = compute_true_Zs(yval)

    algorithms = [
        :AIS, 
        :AISResample,
        :StandardAIS
    ]

    results = Dict{Symbol,Array{Float64,1}}()
    diagnostics = Dict{Symbol,Array{NamedTuple,1}}()
    for name in algorithms
        results[name] = zeros(num_runs)
        diagnostics[name] = Array{NamedTuple,1}(undef, num_runs)
    end

    @threads for i in 1:num_runs
        println("Run $i")
        tabi = TABI(
            AIS(num_samples, num_annealing_dists, SimpleRejection())
        )
        results[:AIS][i], diagnostics[:AIS][i] = estimate(expct_conditioned, tabi)

        tabi_resample = TABI(
            AIS(num_samples, num_annealing_dists, RejectionResample())
        )
        results[:AISResample][i], diagnostics[:AISResample][i] = estimate(
            expct_conditioned, tabi_resample)

        ais = AnnealedISSampler(expct_conditioned.gamma2, num_annealing_dists)
        samples, diag = ais_sample(Random.GLOBAL_RNG, ais, 3*num_samples)
        diag[:samples] = samples
        diagnostics[:StandardAIS][i] = (Z = diag,)
        results[:StandardAIS][i] = AnnealedIS.estimate_expectation(
            samples, x -> x[:x])
    end

    # Save results so they can be used later.
    JLD.save(RESULTS_FILE, "diagnostics", diagnostics, "results", results)

    display_results(results, diagnostics, true_x_posterior_mean, true_Zs)

    convergence_plot(
        diagnostics[:AIS], 
        diagnostics[:AISResample],
        diagnostics[:StandardAIS]
    )
end

main(NUM_ANNEALING_DISTS, NUM_SAMPLES, NUM_RUNS)