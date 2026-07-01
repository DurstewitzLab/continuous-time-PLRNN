using ContPLRNNTraining
using JLD2: @save, @load
using Plots

model_path = "Results/ExampleLorenz"
args = load_json_f32(joinpath(model_path, "args.json"))
# get computing device
device = ContPLRNNTraining.get_device(args)

# dataset
D = load_dataset(args["path_to_data"], args["path_to_time"], args["sequence_time_delta"]; device = device)
#data dimension
N = size(D.X, 2)

# init model
contPLRNN_model = initialize_model(args)

# observation_model
O = initialize_observation_model(args, D)

model_epoch = 2000

#load the model parameters and the state of the model
@load joinpath(model_path, "checkpoints", "epoch_$(model_epoch).jld2") p st

A, W, h = p.cont_model.A, p.cont_model.W, p.cont_model.h
#get dimension of latent space M and number of piecewise linear units P
M, P = length(A), contPLRNN_model.P

#initial state of the model
z0 = init_state(O, D.X[1, :], p.obs_model, st.obs_model)


#length of time series you want to produce
T = 100000.f0#size(D.X, 1)
dt = 1.0f0
tvec = Vector(0.0f0:dt:T)

cache = BatchCache{Any,Any}()

@time z_traj, time_switch_vec, dim_switch_vec = compute_alrnn_trajectory_fully_cached(cache, tvec, A, W, h, z0, P;
save_switching_times = true, max_region_width = 5.0f0)

X_gen = O(z_traj, p.obs_model, st.obs_model)'

#3d plot of trajectory
plot_3D = plot(D.X[:, 1], D.X[:, 2], D.X[:, 3], label = "True trajectory", color = :black)
plot!(plot_3D, X_gen[:, 1], X_gen[:, 2], X_gen[:, 3], label = "Generated trajectory", color = :red)



#plot as a time series
T_transient = 1
T_plot = 1000
max_x = tvec[T_plot]

#plot true trajectory
plot_time_series = plot(D.tvec[T_transient:T_plot], D.X[T_transient:T_plot, :], layout = (N+P, 1), size = (500, 100*(N+P)), color = :black, label = "True trajectory")
#plot generated trajectory with its non-linear units
plot!(plot_time_series, D.tvec[T_transient:T_plot], z_traj[vcat(1:N, M-P+1:M), T_transient:T_plot]', layout = (N+P, 1), color = :red, label = "Generated trajectory")

#plot zero line
hline!(plot_time_series, vcat(repeat([NaN], N), repeat([0.0], P))', layout = (N+P, 1), color = :black, label = "Zero line")

# Add vertical lines for switching times at the appropriate dimensions
for i in 1:length(time_switch_vec)
    switch_time = time_switch_vec[i]
    if switch_time > max_x
        break
    end
    switch_dim = dim_switch_vec[i]
    
    # Map the switching dimension to the corresponding subplot
    subplot_idx = switch_dim + N
    vline!(plot_time_series, [switch_time], subplot = subplot_idx, color = :blue, alpha = 0.7, linewidth = 1, label = "")
end

#add dimension label to the y axis
for i in 1:N
    ylabel!(plot_time_series, "\$x^{($i)}\$", subplot = i)
end
for i in N+1:N+P
    ylabel!(plot_time_series, "\$z^{($i)}\$", subplot = i)
end

#add label to the switching times
vline!(plot_time_series, repeat([NaN], N+P)', layout = (N+P, 1), color= :blue, label = "Switching times")

#show plot again, put legend outside
plot!(plot_time_series, legend = :outertopright, xlim = (D.tvec[T_transient], D.tvec[T_plot]))
plot!(plot_time_series, xlabel = "\$t\$", left_margin = 10Plots.mm,size = (500, 200*(N+P)))


#plot as a time series showing all switching times in the same plot, with different colours 
#for different dimensions
T_transient = 1
T_plot = 200
max_x = tvec[T_plot]

#plot true trajectory
plot_time_series_short = plot(D.tvec[T_transient:T_plot], D.X[T_transient:T_plot, :], layout = (N, 1), size = (500, 200*(N)), color = :black, label = "True trajectory")
#plot generated trajectory with its non-linear units
plot!(plot_time_series_short, D.tvec[T_transient:T_plot], X_gen[T_transient:T_plot, :], layout = (N, 1), color = :red, label = "Generated trajectory")

# Add vertical lines for switching times at the appropriate dimensions
for i in 1:length(time_switch_vec)
    switch_time = time_switch_vec[i]
    if switch_time > max_x
        break
    end
    vline!(plot_time_series_short, repeat([switch_time], N)', color = dim_switch_vec[i], linewidth = 1, label = "")
end

for i in sort(unique(dim_switch_vec))#1:P
    vline!(plot_time_series_short, repeat([NaN], N)', color = i, linewidth = 1, label = "Switching dim $(M-P+i)")
end


plot!(plot_time_series_short, legend = :outertopright, xlim = (D.tvec[T_transient], D.tvec[T_plot]))

#plot(plot_time_series_short)
#plot(plot_time_series)
plot(plot_3D)