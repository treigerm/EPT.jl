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

const FX = :gauss

const RESULTS_FILE = "results.jld"

function compute_true_Zs(yval, f)
    Z1_plus_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval) * max(f(x),0)
    Z1_minus_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval) * (-min(f(x),0))
    Z2_target(x) = pdf(Normal(0, 1), x) * pdf(Normal(x, 1), yval)

    Z1_positive_true, _ = quadgk(Z1_plus_target, 0, 20, atol=1e-16)
    Z1_negative_true, _ = quadgk(Z1_minus_target, -20, 0, atol=1e-16)
    Z2_true, _ = quadgk(Z2_target, -20, 20, atol=1e-16)

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

function convergence_plot(
    taanis_results, 
    taanis_resample_results, 
    ais_results,
    ais_f,
    true_expectation_value
)
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
        ais_intermediates[i,:] = expectations(ais_samples, ais_f)

        Zs = (
            Z1_positive_info = zeros(Float64, num_samples_taanis),
            Z1_negative_info = zeros(Float64, num_samples_taanis),
            Z2_info = Array{Float64}(undef, num_samples_taanis)
        )
        for k in keys(taanis_results[i])
            samples = taanis_results[i][k][:samples]
            Zs[k][:] = normalisation_constants(samples)
        end
        taanis_intermediates[i,:] = (
            Zs[:Z1_positive_info] .- Zs[:Z1_negative_info]) ./ Zs[:Z2_info]
        
        for k in keys(taanis_resample_results[i])
            samples = taanis_resample_results[i][k][:samples]
            Ns = 1:length(samples)
            rejections = taanis_resample_results[i][k][:num_rejected]
            acceptance_ratio = Ns ./ (Ns .+ cumsum(rejections))
            Zs[k][:] = acceptance_ratio .* normalisation_constants(samples)
        end
        taanis_resamples_intermediates[i,:] = (
            Zs[:Z1_positive_info] .- Zs[:Z1_negative_info]) ./ Zs[:Z2_info]
    end 

    ais_errors = relative_squared_error.(
        ais_intermediates, true_expectation_value
    )
    taanis_errors = relative_squared_error.(
        taanis_intermediates, true_expectation_value
    )
    taanis_resamples_errors = relative_squared_error.(
        taanis_resamples_intermediates, true_expectation_value
    )

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
    plot!(
        p,
        1:num_samples_ais,
        fill(true_expectation_value, num_samples_ais),
        linestyle=:dash,
        color=:black,
        label="Ground truth"
    )
    savefig(p, "convergence.png")

    p2 = plot(
        1:num_samples_ais, 
        median(ais_errors, dims=1)[1,:],
        label="AnIS"
    )
    plot!(
        p2, 
        (1:num_samples_taanis) * 3, 
        median(taanis_resamples_errors, dims=1)[1,:],
        label="TAAnIS (resampling)"
    )
    plot!(
        p2, 
        (1:num_samples_taanis) * 3, 
        median(taanis_errors, dims=1)[1,:],
        label="TAAnIS"
    )
    savefig(p2, "errors.png")
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

function main(num_annealing_dists, num_samples, num_runs, fx)
    if fx == :posterior_mean
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x
        end

        yval = 3
        expct_conditioned = expct(yval)

        true_expectation_value = 1.5
        true_Zs = compute_true_Zs(yval, x -> x)

        ais_f = x -> x[:x]
    elseif fx == :seventh_moment
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x^7
        end

        yval = 3
        expct_conditioned = expct(yval)

        posterior_mean = 1.5
        posterior_variance = 0.5

        # Formula taken from https://en.wikipedia.org/wiki/Normal_distribution#Moments
        true_expectation_value = posterior_mean^7 + 
            21*posterior_mean^5*posterior_variance + 
            105*posterior_mean^3*posterior_variance^2 + 
            105*posterior_mean*posterior_variance^3
        true_Zs = compute_true_Zs(yval, x -> x^7)

        ais_f = x -> x[:x]^7
    elseif fx == :gauss
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return pdf(Normal(-y, sqrt(0.5)), x)
        end

        yval = -5
        expct_conditioned = expct(yval)

        # Code from Sheh
        # true_expectation_value = exp(
        #     -0.5 * log(2*π) - 0.5 * (-yval - 0.5 * yval)^2
        # )
        true_expectation_value =  pdf(Normal(yval / 2, 1), -yval)
        true_Zs = compute_true_Zs(
            yval,
            x -> pdf(Normal(-yval, sqrt(0.5)), x)
        )

        ais_f = x -> pdf(Normal(-yval, sqrt(0.5)), x[:x])
    else
        error("Unkown function type: $fx")
    end

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
        tabi = if fx == :gauss
            TABI(
                AIS(num_samples, num_annealing_dists, SimpleRejection()),
                AIS(0, 0, SimpleRejection()),
                AIS(num_samples, num_annealing_dists, SimpleRejection())
            )
        else
            TABI(
                AIS(num_samples, num_annealing_dists, SimpleRejection())
            )
        end
        results[:AIS][i], diagnostics[:AIS][i] = estimate(expct_conditioned, tabi)
        if fx == :gauss
            # Remove Z1_negative_info field because it is not used.
            diagnostics[:AIS][i] = Base.structdiff(
                diagnostics[:AIS][i], (Z1_negative_info=Dict(),)
            )
        end

        tabi_resample = if fx == :gauss
            TABI(
                AIS(num_samples, num_annealing_dists, RejectionResample()),
                AIS(0, 0, RejectionResample()),
                AIS(num_samples, num_annealing_dists, RejectionResample())
            )
        else
            TABI(
                AIS(num_samples, num_annealing_dists, RejectionResample())
            )
        end
        results[:AISResample][i], diagnostics[:AISResample][i] = estimate(
            expct_conditioned, tabi_resample)
        if fx == :gauss
            # Remove Z1_negative_info field because it is not used.
            diagnostics[:AISResample][i] = Base.structdiff(
                diagnostics[:AISResample][i], (Z1_negative_info=Dict(),)
            )
        end

        ais = AnnealedISSampler(expct_conditioned.gamma2, num_annealing_dists)
        samples, diag = ais_sample(Random.GLOBAL_RNG, ais, 3*num_samples)
        diag[:samples] = samples
        diagnostics[:StandardAIS][i] = (Z = diag,)
        results[:StandardAIS][i] = AnnealedIS.estimate_expectation(
            samples, ais_f)
    end

    # Save results so they can be used later.
    JLD.save(RESULTS_FILE, "diagnostics", diagnostics, "results", results)

    display_results(results, diagnostics, true_expectation_value, true_Zs)

    convergence_plot(
        diagnostics[:AIS], 
        diagnostics[:AISResample],
        diagnostics[:StandardAIS],
        ais_f,
        true_expectation_value
    )
end

main(NUM_ANNEALING_DISTS, NUM_SAMPLES, NUM_RUNS, FX)