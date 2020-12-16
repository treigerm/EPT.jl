function collect_log_weights(diagnostics, k)
    num_runs = length(diagnostics)
    num_samples = length(diagnostics[1][k][:samples])

    log_weights = Array{Float64}(undef, num_samples, num_runs)
    for (i, d) in enumerate(diagnostics)
        log_weights[:,i] = map(x -> x.log_weight, d[k][:samples])
    end

    return log_weights
end

function collect_xs(diagnostics, k)
    num_runs = length(diagnostics)
    num_samples = length(diagnostics[1][k][:samples])
    sample = diagnostics[1][k][:samples][1].params

    xs = Array{typeof(sample)}(undef, num_samples, num_runs)
    for (i, d) in enumerate(diagnostics)
        xs[:,i] = map(x -> x.params, d[k][:samples])
    end

    return xs 
end

function ess_batch(log_weights; thinning=1)
    # Input: (num_samples, num_chains)
    # Output: (num_samples/thinning, num_chains)

    num_samples = size(log_weights, 1)
    ess = Array{Float64}(undef, Int(num_samples/thinning), size(log_weights,2))

    ess_ix = 1
    for ix in 1:thinning:num_samples
        denominator = logsumexp(2 * log_weights[1:ix,:]; dims=1)
        numerator = 2 * logsumexp(log_weights[1:ix,:]; dims=1)
        ess[ess_ix,:] = exp.(numerator .- denominator)
        ess_ix += 1
    end
    return ess
end

function get_median_and_quantiles(errors)
    # errors: num_samples x num_chains
    qs = [quantile(errors[i,:], [0.25, 0.75]) for i in 1:size(errors,1)]
    qs = hcat(qs...)
    medians = median(errors, dims=2)[:,1]
    # medians: num_samples
    # qs: 2 x num_samples
    return medians, qs
end

function make_ess_plot(Z1_lws, Z2_lws, anis_lws_retargeted; thinning=1, anis_lws=nothing)
    num_samples = size(Z1_lws, 1)

    @show size(Z1_lws)
    @show size(Z2_lws)
    @show size(anis_lws)
    @show size(anis_lws_retargeted)

    Z1_ess_cumsum = ess_batch(Z1_lws; thinning=thinning)
    Z2_ess_cumsum = ess_batch(Z2_lws; thinning=thinning)

    if isnothing(anis_lws)
        anis_lws = Z2_lws
    end
    anis_ess_cumsum = ess_batch(anis_lws; thinning=thinning)
    anis_ess_retargeted_cumsum = ess_batch(anis_lws_retargeted; thinning=thinning)

    taanis_min_ess = min.(Z1_ess_cumsum, Z2_ess_cumsum)
    anis_min_ess = min.(anis_ess_retargeted_cumsum, anis_ess_cumsum)

    taanis_mds, taanis_qs = get_median_and_quantiles(taanis_min_ess)

    ixs = Array(1:thinning:num_samples)
    ess_plot = plot(
        ixs*2,
        taanis_mds,
        ribbon=[(taanis_mds.-taanis_qs[1,:]), (taanis_qs[2,:].-taanis_mds)],
        xscale=:log10,
        yscale=:log10,
        label="TAAnIS",
        legend=false,
        xlabel="Number of Samples",
        ylabel="ESS",
        xlims=(10,10^4),
        thickness_scaling=1.7
    )
    anis_ixs = Array(1:thinning:2*num_samples)
    anis_mds, anis_qs = get_median_and_quantiles(anis_min_ess)
    plot!(
        ess_plot,
        anis_ixs,
        anis_mds,
        ribbon=[(anis_mds.-anis_qs[1,:]), (anis_qs[2,:].-anis_mds)],
        label="AnIS"
    )
    # plot!(
    #     ess_plot,
    #     ixs,
    #     2*mean(Z2_ess_cumsum; dims=2) ./ (2*ixs),
    #     #ribbon=std(2*Z2_ess_cumsum; dims=2),
    #     label="AnIS"
    # )
    # plot!(
    #     ess_plot,
    #     ixs,
    #     mean(Z2_ess_retargeted_cumsum; dims=2) ./ ixs,
    #     #ribbon=std(Z2_ess_retargeted_cumsum; dims=2),
    #     label="AnIS retargeted"
    # )
    return ess_plot
end

function make_plots(
    result_dir, 
    out_fname; 
    results=nothing, 
    yval=-2,
    dimension=10
)

    yval = yval * ones(dimension) / sqrt(dimension)
    ais_f = x -> pdf(MvNormal(-yval, 0.5*I), x[:x])
    ais_factor = 2

    if isnothing(results)
        results = JLD.load(joinpath(result_dir, "results.jld"))
    end

    diagnostics = results["diagnostics"]
    true_expectation_value = results["true_expectation_value"]

    Z1_lws = collect_log_weights(diagnostics[:AIS], :Z1_positive_info)
    Z2_lws = collect_log_weights(diagnostics[:AIS], :Z2_info)
    anis_xs = collect_xs(diagnostics[:StandardAIS], :Z)
    anis_lws = collect_log_weights(diagnostics[:StandardAIS], :Z)
    anis_lws_retargeted = anis_lws .+ log.(ais_f.(anis_xs))

    ess_plot = make_ess_plot(
        Z1_lws, Z2_lws, anis_lws_retargeted; thinning=25, anis_lws=anis_lws)
    savefig(ess_plot, joinpath(result_dir, "posterior_pred_ess_plots.pdf"))
    return

    p = convergence_plot2(
        diagnostics[:AIS], 
        diagnostics[:StandardAIS],
        ais_f,
        ais_factor,
        true_expectation_value;
        thinning=25
    )
    savefig(p, joinpath(result_dir, out_fname))
end