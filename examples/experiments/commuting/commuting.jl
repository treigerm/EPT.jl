using ExpectationProgramming
using AnnealedIS
using ReverseDiff
using LinearAlgebra: I
using Dates
using Logging
using LoggingExtras
using StatsPlots
using Random

Random.seed!(42)

const RESULTS_FOLDER = "results/"

const NUM_DAYS = 200

const NUM_ANNEALING_DISTS = 100
const NUM_SAMPLES = 100

function make_experiment_folder(experiment_name)
    datestring = Dates.format(Dates.now(), "ddmmyyyy_HHMMSS")
    folder_name = "$(datestring)_$(experiment_name)"
    return mkpath(joinpath(RESULTS_FOLDER, folder_name))
end

function utility(c, t)
    return - (c + 25 * t)
end

function expected_utility(mu, sigma, nu, tau; M=10)
    ts = rand(arraydist([LogNormal(m, s) for (m, s) in zip(mu, sigma)]), M)
    cs = rand(arraydist([LogNormal(m, s) for (m, s) in zip(nu, tau)]), M)
    return mean(utility.(cs, ts); dims=2)
end

function plot_predictive(costs_obs, times_obs, pred_chain)
    num_days = size(costs_obs, 1)
    costs_pred = permutedims(Array(pred_chain[:cost]), [2, 1]) # num_days x num_samples
    times_pred = permutedims(Array(pred_chain[:time]), [2, 1])
    if size(costs_pred, 2) > 20
        ixs = sample(1:size(costs_pred,2), 20, replace=false)
        costs_pred = costs_pred[:,ixs]
        times_pred = times_pred[:,ixs]
    end
    days = 1:num_days
    cp = plot(days, costs_pred, color=:red, alpha=0.5)
    plot!(cp, days, costs_obs, legend=false, color=:black, lw=2)

    tp = plot(days, times_pred, color=:red, alpha=0.5)
    plot!(tp, days, times_obs, legend=false, color=:black, lw=2)
    return cp, tp
end

function generate_data(num_days)
    cycling_cost = 1 
    cycling_time = 10

    walking_cost = 1
    walking_time = 20

    public_transport_cost = 10
    public_transport_time = 5

    cab_cost = 30
    cab_time = 3

    costs = zeros(num_days)
    times = zeros(num_days)
    modes = zeros(Int, num_days)
    for ix in 1:num_days
        mode = rand(DiscreteUniform(1, 4))
        modes[ix] = mode
        if mode == 1
            # Cycling
            costs[ix] = cycling_cost
            times[ix] = cycling_time
        elseif mode == 2
            # Walking
            costs[ix] = walking_cost
            times[ix] = walking_time
        elseif mode == 3
            # Public Transport
            costs[ix] = public_transport_cost
            times[ix] = public_transport_time
        else
            # Cab
            costs[ix] = cab_cost
            times[ix] = cab_time
        end
    end

    return costs, times, modes
end

function main(experiment_name, num_days, num_annealing_dists, num_samples)
    costs, times, modes = generate_data(num_days)

    Turing.setadbackend(:reversediff)

    @expectation function commute(cost, time, d)
        # cost: array length N 
        # time: array length N 
        # d: array length N mode for each trip
        N = length(d)
        num_modes = 4
    
        mu ~ filldist(Normal(0, 1), num_modes)
        sigma ~ filldist(LogNormal(0, 0.01), num_modes)
        if any(sigma .< 0)
            Turing.acclogp!(_varinfo, -Inf)
            return N, N, N, N
        end
        time ~ arraydist([LogNormal(mu[mode], sigma[mode]) for mode in d])
    
        nu ~ filldist(Normal(0, 1), num_modes)
        tau ~ filldist(LogNormal(0, 0.01), num_modes)
        if any(tau .< 0)
            Turing.acclogp!(_varinfo, -Inf)
            return N, N, N, N
        end
        cost ~ arraydist([LogNormal(nu[mode], tau[mode]) for mode in d])  
        u = expected_utility(mu, sigma, nu, tau)
        return u[1], u[2], u[3], u[4]
    end

    result_folder = make_experiment_folder(experiment_name)

    tabi = TABI(
        TuringAlgorithm(AnIS(num_annealing_dists), 0),
        TuringAlgorithm(AnIS(num_annealing_dists), num_samples),
        TuringAlgorithm(AnIS(num_annealing_dists), num_samples)
    )

    turing_model = commute[1].gamma2
    prior_samples = sample(turing_model(missing, missing, modes), Prior(), 1)
    cs = Array(prior_samples[:cost])[1,:]
    ts = Array(prior_samples[:time])[1,:]

    # Get true parameters.
    get_params(chns, param_name) = Array(chns[param_name])[1,:]
    true_mus = get_params(prior_samples, :mu)
    true_sigmas = get_params(prior_samples, :sigma)
    true_nus = get_params(prior_samples, :nu)
    true_taus = get_params(prior_samples, :tau)
    # TODO: Save data in long term format.

    savefig(
        scatter(1:num_days, cs, label="Costs"),
        joinpath(result_folder, "cost.png")
    )
    savefig(
        scatter(1:num_days, ts, label="Times"),
        joinpath(result_folder, "times.png")
    )
    # Plot prior predictive.
    prior_samples = sample(turing_model(cs, ts, modes), Prior(), 10)
    prior_pred = Turing.Inference.predict(
        turing_model(missing, missing, modes),
        prior_samples
    )
    cost_plot, times_plot = plot_predictive(cs, ts, prior_pred)
    savefig(cost_plot, joinpath(result_folder, "prior_pred_cost.png"))
    savefig(times_plot, joinpath(result_folder, "prior_pred_time.png"))

    # TODO: Do AnIS inference.

    # Do NUTS inference for ground truth.
    comm_nuts = sample(turing_model(cs, ts, modes), NUTS(0.65), 4000)

    # Plot posterior predictive.
    post_pred = Turing.Inference.predict(
        turing_model(missing, missing, modes),
        comm_nuts
    )
    cost_plot, times_plot = plot_predictive(cs, ts, post_pred)
    savefig(cost_plot, joinpath(result_folder, "post_pred_cost.png"))
    savefig(times_plot, joinpath(result_folder, "post_pred_time.png"))

    # TODO: Define some metrics how well we can identify the true parameters.

    # NOTE: Just look at one mode of transport for now because they all are the same model.
    # TODO: Plot intermediate weights.
    # TODO: Plot intermediate samples.
    return

    @show true_mus
    @show true_sigmas
    @show true_nus
    @show true_taus
    @show expected_utility(true_mus, true_sigmas, true_nus, true_taus; M=100)

    commute_utilities = [expct(cs, ts, modes) for expct in commute]

    #estimated_utility, chains = estimate(commute_utilities[1], tabi)
    #@show estimated_utility
    estimated_utility, chains = estimate(commute_utilities[2], tabi)
    @show estimated_utility
end

main("test", NUM_DAYS, NUM_ANNEALING_DISTS, NUM_SAMPLES)