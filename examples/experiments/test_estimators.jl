using ExpectationProgramming
using AnnealedIS
using Distributions
using QuadGK
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

include("utils.jl")

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
        ais_factor = 3
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
        ais_factor = 3
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
        ais_factor = 2
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
        samples, diag = ais_sample(Random.GLOBAL_RNG, ais, ais_factor*num_samples)
        diag[:samples] = samples
        diagnostics[:StandardAIS][i] = (Z = diag,)
        results[:StandardAIS][i] = AnnealedIS.estimate_expectation(
            samples, ais_f)
    end

    # Save results so they can be used later.
    JLD.save(
        RESULTS_FILE, 
        "diagnostics", diagnostics, 
        "results", results,
        "true_Zs", true_Zs,
        "true_expectation_value", true_expectation_value,
        "fx", fx,
        "ais_factor", ais_factor
    )

    display_results(results, diagnostics, true_expectation_value, true_Zs)

    convergence_plot(
        diagnostics[:AIS], 
        diagnostics[:AISResample],
        diagnostics[:StandardAIS],
        ais_f,
        ais_factor,
        true_expectation_value
    )
end

main(NUM_ANNEALING_DISTS, NUM_SAMPLES, NUM_RUNS, FX)