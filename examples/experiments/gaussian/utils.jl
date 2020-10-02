using PrettyTables
using Plots
using JLD

Plots.default(lw=2)

const RENAME_ALGORITHMS = Dict(
    "AIS" => "TAAnIS",
    "AISResample" => "TAAnIS (resampling)",
    "StandardAIS" => "AnIS"
)

const RENAME_ESTIMATORS = Dict(
    "Z2_info" => "\$Z_2\$",
    "Z1_positive_info" => "\$Z^+_1\$",
    "Z1_negative_info" => "\$Z^-_1\$",
    "Z" => "\$Z\$"
)

function get_median_and_quantiles(errors)
    qs = [quantile(errors[:,i], [0.25, 0.75]) for i in 1:size(errors,2)]
    qs = hcat(qs...)
    medians = median(errors, dims=1)[1,:]
    return medians, qs
end
function convergence_plot(
    taanis_results, 
    ais_results,
    ais_f,
    ais_factor,
    true_expectation_value;
    suffix=""
)
    num_runs = length(taanis_results)
    num_samples_ais = length(ais_results[1][:Z][:samples])
    # TODO: Control number of samples when Z2 = 0.
    num_samples_taanis = Int(num_samples_ais / ais_factor)
    taanis_intermediates = Array{Float64}(undef, num_runs, num_samples_taanis)
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
    end 

    ais_errors = relative_squared_error.(
        ais_intermediates, true_expectation_value
    )
    taanis_errors = relative_squared_error.(
        taanis_intermediates, true_expectation_value
    )

    p = plot(
        1:num_samples_ais, 
        ais_intermediates[1,:],
        label="AnIS"
    )
    plot!(
        p, 
        (1:num_samples_taanis) * ais_factor, 
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
    savefig(p, "convergence$(suffix).png")

    mds, qs = get_median_and_quantiles(ais_errors) 
    p2 = plot(
        1:num_samples_ais, 
        mds,
        ribbon=[mds.-qs[1,:],qs[2,:]-mds],
        label="AnIS",
        xscale=:log10,
        yscale=:log10
    )
    mds, qs = get_median_and_quantiles(taanis_errors) 
    plot!(
        p2, 
        (1:num_samples_taanis) * ais_factor, 
        mds,
        ribbon=[mds.-qs[1,:],qs[2,:]-mds],
        label="TAAnIS"
    )
    xlabel!(p2, "Number of Samples")
    ylabel!(p2, "Relative Squared Error")
    savefig(p2, "errors$(suffix).png")

    return p, p2
end

function relative_squared_error(x_hat, true_x) 
    return (x_hat - true_x)^2 / true_x^2
end

function display_results(
    logger, 
    results, # Dict with alg_name => expectation_estimates
    diagnostics, # Dict with alg_name => diagnostics
    true_Z, # True value for expectation
    true_Zs # True values of the individual normalisation constants for TABI
)
    with_logger(logger) do
        for (name, v) in pairs(results)
            squared_errors = relative_squared_error.(v, true_Z)
            mean_error = mean(squared_errors)
            @info "$name:"
            @info ""
            @info "Values: $v"
            @info "Relative squared errors: $squared_errors"
            @info "Mean relative squared errors: $mean_error"

            diags = diagnostics[name] # Array of NamedTuple
            @info "ESS:"
            for estimator_name in keys(diags[1])
                ess_mean = mean(map(x -> x[estimator_name][:ess], diags))
                @info "$estimator_name: $ess_mean"
            end

            if name in [:AIS, :AISResample]
                @info "Error in individual estimates:"
                for estimator_name in keys(diags[1])
                    errors = map(diags) do Z_est
                        relative_squared_error(
                            Z_est[estimator_name][:Z_estimate], 
                            true_Zs[estimator_name]
                        )
                    end
                    error_mean = mean(errors)
                    
                    @info "$estimator_name: $error_mean"
                end
            end
            @info "------------------------------"
        end
    end
end

"""
Diagnostics is a dict:

Dict(
    alg_name1 => [
        (
        estimator_name1 = Dict(
                :ess => Float,
                :Z_estimate => Float,
                (:intermediate_samples => Array)
            ),
        estimator_name2 = Dict(...),
        ...
        ),
        (
        different_estimator_name = Dict(...)
        )
    ],
    alg_name2 => [...],
    ...
)

Want to convert it into table with:

alg_name | estimator_name1 | estimator_name2 | different_estimator_name
"""
function make_latex_table_ess(diagnostics)
    # Get column names.
    columns = String[]
    for (name, diag_runs) in pairs(diagnostics) 
        for n in keys(diag_runs[1])
            push!(columns, string(n))
        end
    end
    unique!(columns)

    num_rows = length(keys(diagnostics))
    num_cols = length(columns)
    alg_names = Array{String,1}(undef, num_rows)
    values = Array{String,2}(undef, num_rows, num_cols)
    fill!(values, "N/A")

    for (row_ix, (name, diag_runs)) in enumerate(pairs(diagnostics))
        alg_names[row_ix] = string(name)
        for n in keys(diag_runs[1])
            ess_mean = round(mean(map(x -> x[n][:ess], diag_runs)), digits=2)
            ess_std = round(std(map(x -> x[n][:ess], diag_runs)), digits=2)
            col_ix = findall(x -> x == string(n), columns)[1]
            values[row_ix,col_ix] = "\$$(ess_mean) \\pm $(ess_std)\$"
            #ess_quantiles = round.(quantile(
            #    map(x -> x[n][:ess], diag_runs), [0.25, 0.75]
            #), digits=2)
            #values[row_ix,col_ix] = "\$$(ess_quantiles[1]) \\leq $(ess_median) \\leq $(ess_quantiles[2])\$"
        end
    end

    alg_names = map(x -> RENAME_ALGORITHMS[x], alg_names)
    t = hcat(alg_names, values)
    columns = map(x -> RENAME_ESTIMATORS[x], columns)
    pushfirst!(columns, "Algorithm")

    pretty_table(t, columns, backend=:latex)
end

function make_latex_table_log_errors(results, diagnostics, true_Zs, true_Z)
    true_Zs = Dict(pairs(true_Zs))
    true_Zs[:Z] = true_Z

    # Get column names.
    columns = String["Z2_info", "Z1_positive_info", "Z1_negative_info"]

    num_rows = length(keys(diagnostics))
    num_cols = length(columns)
    alg_names = Array{String,1}(undef, num_rows)
    Z_errors = Array{String,1}(undef, num_rows)
    values = Array{String,2}(undef, num_rows, num_cols)
    fill!(values, "N/A")

    for (row_ix, (name, diag_runs)) in enumerate(pairs(diagnostics))
        alg_names[row_ix] = string(name)

        alg_Z_errors = log.(relative_squared_error.(results[name], true_Z))
        alg_Z_error_mean = round(mean(alg_Z_errors), digits=2)
        alg_Z_error_std = round(std(alg_Z_errors), digits=2)
        Z_errors[row_ix] = "\$$(alg_Z_error_mean) \\pm $(alg_Z_error_std)\$"

        if name == :StandardAIS
            continue
        end 

        for n in keys(diag_runs[1])
            errors = map(diag_runs) do Z_est
                log(relative_squared_error(
                    Z_est[n][:Z_estimate], 
                    true_Zs[n]
                ))
            end
            error_mean = round(mean(errors), digits=2)
            error_std = round(std(errors), digits=2)
            col_ix = findall(x -> x == string(n), columns)[1]
            values[row_ix,col_ix] = "\$$(error_mean) \\pm $(error_std)\$"
        end
    end

    alg_names = map(x -> RENAME_ALGORITHMS[x], alg_names)
    t = hcat(alg_names, Z_errors, values)
    columns = map(x -> RENAME_ESTIMATORS[x], columns)
    pushfirst!(columns, "\$Z\$")
    pushfirst!(columns, "Algorithm")

    pretty_table(t, columns, backend=:latex)
end

function get_latex_tables(filename)
    d = load(filename)
    true_Zs, true_expectation_value = d["true_Zs"], d["true_expectation_value"]
    results, diagnostics = d["results"], d["diagnostics"]
    fx = d["fx"]

    println("Effective Sample Size:\n")
    make_latex_table_ess(diagnostics)
    println("")
    println("Log relative squared error:\n")
    make_latex_table_log_errors(
        results, diagnostics, true_Zs, true_expectation_value
    )
end

function load_and_analyse(filename; suffix=nothing)
    d = load(filename)
    true_Zs, true_expectation_value = d["true_Zs"], d["true_expectation_value"]
    results, diagnostics = d["results"], d["diagnostics"]
    fx = d["fx"]
    ais_factor = if haskey(d, "ais_factor")
        d["ais_factor"]
    else
        3 # This is left to be backwards compatible. Old result files didn't store the ais_factor key.
    end
    if isnothing(suffix)
        suffix = "_$(string(fx))"
    end

    ais_f = if fx == :posterior_mean
        x -> x[:x]
    elseif fx == :seventh_moment
        x -> x[:x]^7
    else
        x -> pdf(Normal(5, sqrt(0.5)), x[:x])
    end

    display_results(results, diagnostics, true_expectation_value, true_Zs)

    convergence_plot(
        diagnostics[:AIS], 
        diagnostics[:AISResample],
        diagnostics[:StandardAIS],
        ais_f,
        ais_factor,
        true_expectation_value;
        suffix=suffix
    )
end

function make_experiment_folder(results_folder, experiment_name)
    datestring = Dates.format(Dates.now(), "ddmmyyyy_HHMMSS")
    folder_name = "$(datestring)_$(experiment_name)"
    return mkpath(joinpath(results_folder, folder_name))
end