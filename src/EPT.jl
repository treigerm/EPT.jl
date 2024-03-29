module EPT

using Reexport
@reexport using Turing
using AnnealedIS
using Random

import MacroTools

export @expectation,
    Expectation,
    TABI,
    AIS,
    TuringAlgorithm,
    estimate_expectation,
    # Reexports
    RejectionResample,
    SimpleRejection

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

macro expectation(expr)
    fn_dict = MacroTools.splitdef(expr)

    fn_name = fn_dict[:name]
    expct2_name = gensym(fn_name)

    fn_dict[:name] = expct2_name
    fn_expr_expct2 = MacroTools.combinedef(fn_dict)
    fn_expr_expct2 = Turing.DynamicPPL.model(fn_expr_expct2, false)

    fn_body = fn_dict[:body]
    pos_bodies = translate_return(fn_body, true)
    neg_bodies = translate_return(fn_body, false)

    num_expectations = length(pos_bodies)
    expressions = Array{Expr}(undef, num_expectations + 2)
    arr_name = gensym()
    expressions[1] = quote
        $fn_expr_expct2
    
        $(arr_name) = Array{Expectation}(undef, $num_expectations)
    end 

    for i in 1:num_expectations
        expct1_pos_name = gensym(fn_name)
        expct1_neg_name = gensym(fn_name)

        fn_dict[:body] = pos_bodies[i]
        fn_dict[:name] = expct1_pos_name
        fn_expr_expct1_pos = MacroTools.combinedef(fn_dict)
        fn_expr_expct1_pos = Turing.DynamicPPL.model(fn_expr_expct1_pos, false)

        fn_dict[:body] = neg_bodies[i]
        fn_dict[:name] = expct1_neg_name
        fn_expr_expct1_neg = MacroTools.combinedef(fn_dict)
        fn_expr_expct1_neg = Turing.DynamicPPL.model(fn_expr_expct1_neg, false)

        expressions[i + 1] = quote
            $fn_expr_expct1_pos
    
            $fn_expr_expct1_neg
    
            $(arr_name)[$i] = Expectation(
                $(expct1_pos_name),
                $(expct1_neg_name),
                $(expct2_name)
            )
        end
    end

    if num_expectations == 1
        expressions[end] = quote
            $(fn_name) = $(arr_name)[1]
        end
    else
        expressions[end] = quote 
            $(fn_name) = $(arr_name)
        end
    end

    final_expr = Expr(:block, expressions...)
    
    return esc(final_expr)
end

function translate_return(expr, is_positive_expectation)
    num_expectations = 0
    # NOTE: We assume that all return statements have the same number of returns
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, return r_)
            if isa(r, Expr) && r.head == :tuple
                num_expectations = length(r.args)
            else
                num_expectations = 1
            end
        else
            e
        end
    end

    expressions = Array{Expr}(undef, num_expectations)
    for i in 1:num_expectations
        expressions[i] = MacroTools.postwalk(expr) do e
            if MacroTools.@capture(e, return r_)
                tmp_var = gensym()
                if num_expectations > 1
                    r = r.args[i]
                end
                # NOTE: The compiler is smart enough to remove the if condition.
                # TODO: Possibly use the macro @addlogp!() to get rid of warning
                quote
                    $(tmp_var) = $r
                    if _context isa Turing.DefaultContext
                        if $is_positive_expectation
                            Turing.acclogp!(_varinfo, log(max($(tmp_var), 0)))
                        else
                            Turing.acclogp!(_varinfo, log(-min($(tmp_var), 0)))
                        end
                    end
                    return $(tmp_var)
                end
            else
                e
            end
        end
    end

    # Return an array of expressions
    return expressions
end

struct TABI{S <: Turing.InferenceAlgorithm,T <: Turing.InferenceAlgorithm,U <: Turing.InferenceAlgorithm}
    Z1_pos_alg::S
    Z1_neg_alg::T
    Z2_alg::U
end

function TABI(marginal_likelihood_estimator::T) where {T <: Turing.InferenceAlgorithm}
    return TABI(
        marginal_likelihood_estimator,
        marginal_likelihood_estimator,
        marginal_likelihood_estimator 
    )
end

# Make marginal_likelihood_estimator a keyword arg.
function TABI(;marginal_likelihood_estimator::T) where {T <: Turing.InferenceAlgorithm}
    return TABI(
        marginal_likelihood_estimator,
        marginal_likelihood_estimator,
        marginal_likelihood_estimator 
    )
end

function Base.show(io::IO, tabi::TABI)
    s = "TABI(\n\t$(tabi.Z1_pos_alg),\n\t$(tabi.Z1_neg_alg),\n\t$(tabi.Z2_alg)\n)"
    print(io, s)
end

struct AIS{T <: AnnealedIS.RejectionSampler} <: Turing.InferenceAlgorithm
    num_samples::Int
    num_annealing_dists::Int
    rejection_sampler::T
end

struct TuringAlgorithm{T <: Turing.InferenceAlgorithm} <: Turing.InferenceAlgorithm
    inference_algorithm::T
    num_samples::Int
end

function estimate_expectation(
    expct::Expectation, 
    alg::TABI{T,T,T}; 
    kwargs...
) where {T <: AIS}
    ais = AnnealedISSampler(
        expct.gamma2, 
        alg.Z2_alg.num_annealing_dists,
        alg.Z2_alg.rejection_sampler
    )
    prior_density = ais.prior_density
    Z2_diagnostics = estimate_normalisation_constant(
        ais,
        alg.Z2_alg.num_samples;
        kwargs...
    )
    Z2 = Z2_diagnostics[:Z_estimate]

    if alg.Z1_pos_alg.num_samples > 0
        ais = AnnealedISSampler(
            rng -> AnnealedIS.sample_from_prior(rng, expct.gamma1_pos), 
            prior_density,
            AnnealedIS.make_log_joint_density(expct.gamma1_pos),
            alg.Z1_pos_alg.num_annealing_dists,
            alg.Z1_pos_alg.rejection_sampler
        )
        Z1_positive_diagnostics = estimate_normalisation_constant(
            ais,
            alg.Z1_pos_alg.num_samples;
            kwargs...
        )
        Z1_positive = Z1_positive_diagnostics[:Z_estimate]
    else
        Z1_positive = 0
        Z1_positive_diagnostics = Dict()
        Z1_negative_diagnostics[:Z_estimate] = Z1_positive
    end

    if alg.Z1_neg_alg.num_samples > 0
        ais = AnnealedISSampler(
            rng -> AnnealedIS.sample_from_prior(rng, expct.gamma1_neg), 
            prior_density,
            AnnealedIS.make_log_joint_density(expct.gamma1_neg),
            alg.Z1_neg_alg.num_annealing_dists,
            alg.Z1_neg_alg.rejection_sampler
        )
        Z1_negative_diagnostics = estimate_normalisation_constant(
            ais,
            alg.Z1_neg_alg.num_samples;
            kwargs...
        )
        Z1_negative = Z1_negative_diagnostics[:Z_estimate]
    else
        Z1_negative = 0
        Z1_negative_diagnostics = Dict()
        Z1_negative_diagnostics[:Z_estimate] = Z1_negative
    end

    return (Z1_positive - Z1_negative) / Z2, (
        Z2_info = Z2_diagnostics,
        Z1_positive_info = Z1_positive_diagnostics,
        Z1_negative_info = Z1_negative_diagnostics,
    )
end

function estimate_expectation(
    expct::Expectation,
    alg::TABI{T,T,T};
    kwargs...
) where {T <: TuringAlgorithm}
    Z1_positive, Z1_positive_chain = estimate_normalisation_constant(
        expct.gamma1_pos, alg.Z1_pos_alg; kwargs...
    )

    Z1_negative, Z1_negative_chain = estimate_normalisation_constant(
        expct.gamma1_neg, alg.Z1_neg_alg; kwargs...
    )

    Z2, Z2_chain = estimate_normalisation_constant(
        expct.gamma2, alg.Z2_alg; kwargs...
    )

    return (Z1_positive - Z1_negative) / Z2, (
        Z2_info = Z2_chain,
        Z1_positive_info = Z1_positive_chain,
        Z1_negative_info = Z1_negative_chain,
    )
end

function estimate_normalisation_constant(
    model::Turing.Model,
    alg::TuringAlgorithm;
    kwargs...
)
    Z_estimate = 0.0
    chain = nothing
    if alg.num_samples > 0
        chain = sample(
            model, 
            alg.inference_algorithm, 
            alg.num_samples;
            kwargs...
        )
        Z_estimate = exp(chain.logevidence)
    end
    return Z_estimate, chain
end

function estimate_normalisation_constant(
    ais::AnnealedISSampler{SimpleRejection},
    num_samples;
    kwargs...
)
    samples, diagnostics = ais_sample(
        Random.GLOBAL_RNG, 
        ais, 
        num_samples;
        kwargs...
    )
    diagnostics[:samples] = samples
    diagnostics[:Z_estimate] = normalisation_constant(samples)
    return diagnostics
end

function estimate_normalisation_constant(
    ais::AnnealedISSampler{RejectionResample},
    num_samples;
    kwargs...
)
    num_samples = num_samples
    samples, diagnostics = ais_sample(
        Random.GLOBAL_RNG, 
        ais, 
        num_samples;
        kwargs...
    )
    diagnostics[:samples] = samples
    Z_est = normalisation_constant(samples)
    acceptance_ratio = num_samples / (num_samples + sum(diagnostics[:num_rejected]))
    diagnostics[:Z_estimate] = acceptance_ratio * Z_est
    return diagnostics
end

function normalisation_constant(samples)
    sum_weights = sum(samples) do weighted_sample
        exp(weighted_sample.log_weight)
    end
    return sum_weights / length(samples)
end

end
