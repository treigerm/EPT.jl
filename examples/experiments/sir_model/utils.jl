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
    dist_ixs = [i > 10 ? (i-9)*10 : i for i in ixs]
    betas = [betas[i] for i in dist_ixs]
    ess = ess[ixs]
    return plot(
        dist_ixs,
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

function est_exp(chn, f)
    log_weights = Array(chn[:log_weight])
    i₀ = Array(chn[:i₀])
    β = Array(chn[:β])

    normalisation = exp(logsumexp(log_weights))
    weighted_sum = sum(zip(log_weights, i₀, β)) do (lw, i, b)
        exp(lw) * f(i, b)
    end

    return weighted_sum / normalisation
end

function process_results(
    result_folder; 
    ode_tabi, 
    R0_estimate,
    true_expectation_value,
    tabi,
    X,
    Y,
    obstimes,
    i_0_true,
    β_true,
    cost_fn
)
    logger = TeeLogger(
        ConsoleLogger(), 
        FileLogger(joinpath(result_folder, "out.log"))
    )
    betas = tabi.Z2_alg.inference_algorithm.betas

    """
    savefig(plot_intermediate_samples(
        ode_tabi[:Z2_info].info[:intermediate_samples], :β, β_true, betas),
        joinpath(result_folder, "intermediate_samples_beta.png")
    )
    savefig(plot_intermediate_samples(
        ode_tabi[:Z2_info].info[:intermediate_samples], :i₀, i_0_true, betas),
        joinpath(result_folder, "intermediate_samples_i_0.png")
    )
    """
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

    anis_estimate = est_exp(ode_tabi[:Z2_info], cost_fn)

    with_logger(logger) do
        @info "True expectation value: $(true_expectation_value)"
        @info "Expectation estimate TABI: $(R0_estimate)"
        @info "Expectation estimate AnIS: $(anis_estimate)"

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

function numeric_expectation_calculation(model, f)
    i_max = 1_000
	beta_max = 20
    params_max = [i_max, beta_max]

    spl = Turing.Sampler(Turing.NUTS(0.65), model)
    logjoint = AnnealedIS.gen_logjoint(spl.state.vi, model, spl)

    norm_const_evals = Float64[]
    function norm_const_fn(x, p)
        lj = logjoint(x)
        push!(norm_const_evals, lj)
        return exp(lj)
    end
    
    norm_solve = solve(
		QuadratureProblem(norm_const_fn, zeros(2), params_max),
		HCubatureJL(), reltol=1e-3, abstol=1e-3
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
		HCubatureJL(), reltol=1e-3, abstol=1e-3
	)
    exp_gamma_f = unnorm_exp_solve.u
    exp_gamma_f_logsumexp = logsumexp(unnorm_exp_evals)

    return exp(exp_gamma_f_logsumexp - Z_logsumexp)
end