using ExpectationProgramming
using AnnealedIS
using Distributions
using QuadGK
using JLD
using LoggingExtras
using Logging
using Dates

using Random
using Base.Threads
using Statistics: mean
using LinearAlgebra: I

const NUM_ANNEALING_DISTS = 100
const NUM_SAMPLES = 100
const NUM_RUNS = 10

const FX = :gauss
const Y_OBSERVED = -2
const DIMENSION = 10

const RANDOM_SEED = 42

const EXPERIMENT_NAME = "test_intermediate_$(RANDOM_SEED)_samples$(NUM_SAMPLES)_y$(-Y_OBSERVED)_D$(DIMENSION)"
const RESULTS_FOLDER = "/data/reichelt/tabi/gaussian"
const RESULTS_FILE = "results.jld"

Random.seed!(RANDOM_SEED) 
 
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

function main(
    experiment_name, 
    num_annealing_dists, 
    num_samples, 
    num_runs, 
    fx,
    yval,
    dimension
)
    result_folder = make_experiment_folder(RESULTS_FOLDER, experiment_name)
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, "out.log"))
    )

    @expectation function expct(y)
        x ~ MvNormal(zeros(length(y)), I) 
        y ~ MvNormal(x, I)
        #return pdf(MvNormal(-y, sqrt(0.5)*I), x)
        return pdf(MvNormal(-y, 0.5*I), x)
    end

    yval = yval * ones(dimension) / sqrt(dimension)
    expct_conditioned = expct(yval)

    # Code from Sheh
    # true_expectation_value = exp(
    #     -0.5 * log(2*π) - 0.5 * (-yval - 0.5 * yval)^2
    # )
    true_expectation_value = pdf(MvNormal(yval / 2, I), -yval)
    true_Zs = compute_true_Zs(
        yval[1],
        x -> pdf(Normal(-yval[1], sqrt(0.5)), x)
    )

    #ais_f = x -> pdf(MvNormal(-yval, sqrt(0.5)*I), x[:x])
    ais_f = x -> pdf(MvNormal(-yval, 0.5*I), x[:x])
    ais_factor = 2

    algorithms = [
        :AIS, 
        :StandardAIS
    ]

    results = Dict{Symbol,Array{Float64,1}}()
    diagnostics = Dict{Symbol,Array{NamedTuple,1}}()
    for name in algorithms
        results[name] = zeros(num_runs)
        diagnostics[name] = Array{NamedTuple,1}(undef, num_runs)
    end

    #@threads for i in 1:num_runs
    @time begin
    for i in 1:num_runs
        println("Run $i")
        tabi = TABI(
            AIS(num_samples, num_annealing_dists, SimpleRejection()),
            AIS(0, 0, SimpleRejection()),
            AIS(num_samples, num_annealing_dists, SimpleRejection())
        )
        results[:AIS][i], diagnostics[:AIS][i] = estimate(
            expct_conditioned, 
            tabi;
            store_intermediate_samples=true
        )
        # Remove Z1_negative_info field because it is not used.
        diagnostics[:AIS][i] = Base.structdiff(
            diagnostics[:AIS][i], (Z1_negative_info=Dict(),)
        )

        ais = AnnealedISSampler(expct_conditioned.gamma2, num_annealing_dists)
        samples, diag = ais_sample(Random.GLOBAL_RNG, ais, ais_factor*num_samples)
        diag[:samples] = samples
        diagnostics[:StandardAIS][i] = (Z = diag,)
        results[:StandardAIS][i] = AnnealedIS.estimate_expectation(
            samples, ais_f
        )
    end
    end

    # Save results so they can be used later.
    JLD.save(
        joinpath(result_folder, RESULTS_FILE),
        "diagnostics", diagnostics, 
        "results", results,
        "true_Zs", true_Zs,
        "true_expectation_value", true_expectation_value,
        "fx", fx,
        "ais_factor", ais_factor
    )

    display_results(logger, results, diagnostics, true_expectation_value, true_Zs)

    conv_plot, error_plot = convergence_plot(
        diagnostics[:AIS], 
        diagnostics[:StandardAIS],
        ais_f,
        ais_factor,
        true_expectation_value
    )
    savefig(conv_plot, joinpath(result_folder, "convergence.png"))
    savefig(error_plot, joinpath(result_folder, "errors.png"))
end

main(
    EXPERIMENT_NAME, 
    NUM_ANNEALING_DISTS, 
    NUM_SAMPLES, 
    NUM_RUNS, 
    FX, 
    Y_OBSERVED,
    DIMENSION
)