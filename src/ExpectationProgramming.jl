module ExpectationProgramming

# NOTE: Things don't work if we do "import Turing" because of some scoping 
# behaviour in the macros.
using Turing
using AnnealedIS
using Random

import MacroTools

export @expectation,
    Expectation,
    TABI,
    AIS,
    estimate

# TODO: What are the correct types for the fields? They are Turing models.
# They can either be a turing model or a turing modelgen.
struct Expectation
    gamma1_pos
    gamma1_neg
    gamma2
end

function (expct::Expectation)(args...; kwargs...)
    # TODO: Check that expct contains Turing.ModelGen's.
    return Expectation(
        expct.gamma1_pos(args...; kwargs...),
        expct.gamma1_neg(args...; kwargs...),
        expct.gamma2(args...; kwargs...)
    )
end

# TODO: Handle multiple return values, e.g. return a, b, c
macro expectation(expr)
    fn_dict = MacroTools.splitdef(expr)

    fn_name = fn_dict[:name]
    expct1_pos_name = gensym(fn_name)
    expct1_neg_name = gensym(fn_name)
    expct2_name = gensym(fn_name)

    fn_dict[:name] = expct2_name
    fn_expr_expct2 = MacroTools.combinedef(fn_dict)

    fn_body = fn_dict[:body]
    fn_dict[:body] = translate_return(fn_body, true)
    fn_dict[:name] = expct1_pos_name
    fn_expr_expct1_pos = MacroTools.combinedef(fn_dict)

    fn_dict[:body] = translate_return(fn_body, false)
    fn_dict[:name] = expct1_neg_name
    fn_expr_expct1_neg = MacroTools.combinedef(fn_dict)

    return quote
        esc(Turing.@model($fn_expr_expct1_pos))

        esc(Turing.@model($fn_expr_expct1_neg))

        esc(Turing.@model($fn_expr_expct2))

        $(esc(fn_name)) = Expectation(
            $(expct1_pos_name),
            $(expct1_neg_name),
            $(expct2_name)
        )
    end
end

function translate_return(expr, is_positive_expectation)
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, return r_)
            tmp_var = gensym()
            # NOTE: The compiler is smart enough to remove the if condition.
            # TODO: Possibly use the macro @addlogp!() to get rid of warning
            quote
                $(tmp_var) = $r
                if $is_positive_expectation
                    Turing.acclogp!(_varinfo, log(max($(tmp_var), 0)))
                else
                    Turing.acclogp!(_varinfo, log(-min($(tmp_var), 0)))
                end
                return $(tmp_var)
            end
        else
            e
        end
    end
end

struct TABI{T<:Turing.InferenceAlgorithm}
    estimation_alg::T
end

struct AIS <: Turing.InferenceAlgorithm
    num_samples::Int
    num_annealing_dists::Int
end

# TODO: Check the Distributions package and how estimate is used there. 
# Potentially rename this function.
function Distributions.estimate(expct::Expectation, alg::TABI{AIS})
    ais = AnnealedISSampler(expct.gamma2, alg.estimation_alg.num_annealing_dists)
    prior_density = ais.prior_density
    samples = ais_sample(Random.GLOBAL_RNG, ais, alg.estimation_alg.num_samples)
    Z2 = normalisation_constant(samples)

    ais = AnnealedISSampler(
        rng -> AnnealedIS.sample_from_prior(rng, expct.gamma1_pos), 
        prior_density,
        AnnealedIS.make_log_joint_density(expct.gamma1_pos),
        alg.estimation_alg.num_annealing_dists
    )
    samples = ais_sample(Random.GLOBAL_RNG, ais, alg.estimation_alg.num_samples)
    Z1_positive = normalisation_constant(samples)

    ais = AnnealedISSampler(
        rng -> AnnealedIS.sample_from_prior(rng, expct.gamma1_neg),
        prior_density,
        AnnealedIS.make_log_joint_density(expct.gamma1_neg),
        alg.estimation_alg.num_annealing_dists
    )
    samples = ais_sample(Random.GLOBAL_RNG, ais, alg.estimation_alg.num_samples)
    Z1_negative = normalisation_constant(samples)

    return (Z1_positive - Z1_negative) / Z2
end

function estimate_normalisation_constant(model::Turing.Model, alg::AIS)
    ais = AnnealedISSampler(model, alg.num_annealing_dists)
    # TODO: Have nicer way to pass in rng.
    samples = ais_sample(Random.GLOBAL_RNG, ais, alg.num_samples)
    return normalisation_constant(samples)
end

function normalisation_constant(samples)
    sum_weights = sum(samples) do weighted_sample
        exp(weighted_sample.log_weight)
    end
    return sum_weights / length(samples)
end

end
