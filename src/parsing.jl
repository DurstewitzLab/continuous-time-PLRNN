using ArgParse
using Lux, LuxCore
using Lux: Chain, Dense, glorot_uniform
using Flux: cpu, gpu
using Optimisers

using ..Utilities


function initialize_model(args::AbstractDict; mod = @__MODULE__)
    # gather args
    M = args["latent_dim"]
    P = args["pwl_units"]
    id_tf = args["observation_model"] == "Identity"

    model_name = args["model"]

    # model type in correct module scope
    model_t = @eval mod $(Symbol(model_name))

    # solver hyperparameters travel with the model
    hyper_config = ALRNNHyperConfig(args)

    # initialize model
    if model_t <: contALRNN
        model = model_t(M, P; hyper_config)
    elseif model_name == "contPLRNN"
        model = contALRNN(M, M; hyper_config)
    else
        @error("Model $model_name is not implemented!")
    end

    println("Model / # Parameters: $(model) / $(LuxCore.parameterlength(model))")
    return model
end

function initialize_observation_model(args::AbstractDict, D::AbstractDataset)
    N = size(D.X, 2)
    M = args["latent_dim"]

    # initialize by default w/o bias
    if args["observation_model"] == "Identity"
        obs_model = Identity(N, M)
    else
        @error("Model $(args["observation_model"]) is not implemented!")
    end

    #println("Obs. Model / # Parameters: $(typeof(obs_model)) / $(parameterlength(obs_model))")
    return obs_model
end

function initialize_optimizer(args::Dict{String, Any})
    # optimizer chain
    ηₛ = args["start_lr"]::Float32
    κ = args["gradient_clipping_norm"]::Float32
    
    # Create optimizer chain
    opt_vec = []
    
    # set gradient clipping
    if κ > zero(κ)
        push!(opt_vec, Optimisers.ClipNorm(κ)) 
    end

    # set SGD optimizer (ADAM, RADAM, etc)
    opt_sym = Symbol(args["optimizer"])
    opt = @eval $opt_sym($ηₛ)
    push!(opt_vec, opt)

    # Return optimizer chain if we have multiple components, otherwise just the optimizer
    if length(opt_vec) > 1
        return Optimisers.OptimiserChain(opt_vec...)
    else
        return opt_vec[1]
    end
end

get_device(args::AbstractDict) = @eval $(Symbol(args["device"]))

"""
    argtable()

Prepare the argument table holding the information of all possible arguments
and correct datatypes.
"""
function argtable()
    settings = ArgParseSettings()
    defaults = load_defaults()

    @add_arg_table! settings begin
        # meta
        "--experiment"
        help = "The overall experiment name."
        arg_type = String
        default = defaults["experiment"] |> String

        "--name"
        help = "Name of a single experiment instance."
        arg_type = String
        default = defaults["name"] |> String

        "--run", "-r"
        help = "The run ID."
        arg_type = Int
        default = defaults["run"] |> Int

        "--saving_interval"
        help = "The interval at which scalar quantities are stored measured in epochs."
        arg_type = Int
        default = defaults["saving_interval"] |> Int

        # data
        "--path_to_data", "-d"
        help = "Path to dataset used for training."
        arg_type = String
        default = defaults["path_to_data"] |> String

        # time points
        "--path_to_time"
        help = "Path to time points of dataset used for training."
        arg_type = String
        default = defaults["path_to_time"] |> String

        # training
        "--teacher_forcing_interval"
        help = "The teacher forcing interval to use."
        arg_type = Int
        default = defaults["teacher_forcing_interval"] |> Int

        "--gaussian_noise_level"
        help = "Noise level of gaussian noise added to teacher signals."
        arg_type = Float32
        default = defaults["gaussian_noise_level"] |> Float32

        "--sequence_length", "-T"
        help = "Length of sequences sampled from the dataset during training."
        arg_type = Int
        default = defaults["sequence_length"] |> Int

        "--sequence_time_delta"
        help = "Integration time delta for training sequences."
        arg_type = Float32
        default = defaults["sequence_time_delta"] |> Float32

        "--batch_size", "-S"
        help = "The number of sequences to pack into one batch."
        arg_type = Int
        default = defaults["batch_size"] |> Int

        "--epochs", "-e"
        help = "The number of epochs to train for."
        arg_type = Int
        default = defaults["epochs"] |> Int

        "--batches_per_epoch" 
        help = "The number of batches processed in each epoch."
        arg_type = Int
        default = defaults["batches_per_epoch"] |> Int

        "--gradient_clipping_norm"
        help = "The norm at which to clip gradients during training."
        arg_type = Float32
        default = defaults["gradient_clipping_norm"] |> Float32

        "--optimizer"
        help = "The optimizer to use for SGD optimization. Must be one provided by Flux.jl."
        arg_type = String
        default = defaults["optimizer"] |> String

        "--start_lr"
        help = "Learning rate passed to the optimizer at the beginning of training."
        arg_type = Float32
        default = defaults["start_lr"] |> Float32

        "--end_lr"
        help = "Target learning rate at the end of training due to exponential decay."
        arg_type = Float32
        default = defaults["end_lr"] |> Float32

        "--device"
        help = "Training device to use."
        arg_type = String
        default = defaults["device"] |> String

        "--BLAS_threads"
        help = "Number of threads to use for BLAS."
        arg_type = Int
        default = defaults["BLAS_threads"] |> Int

        # model
        "--model", "-m"
        help = "RNN to use."
        arg_type = String
        default = defaults["model"] |> String

        "--latent_dim", "-M"
        help = "RNN latent dimension."
        arg_type = Int
        default = defaults["latent_dim"] |> Int

        "--pwl_units", "-P"
        help = "Number of piecewise linear units."
        arg_type = Int
        default = defaults["pwl_units"] |> Int

        "--pretrained_path"
        help = "Path to pretrained model. Leave empty string to train from scratch."
        arg_type = String
        default = defaults["pretrained_path"] |> String

        "--observation_model", "-o"
        help = "Observation model to use."
        arg_type = String
        default = defaults["observation_model"] |> String

        "--lat_model_regularization"
        help = "Regularization λ for latent model parameters."
        arg_type = Float32
        default = defaults["lat_model_regularization"] |> Float32

        "--A_matrix_regularization"
        help = "Regularization λ for A matrix parameters (penalizes positive values)."
        arg_type = Float32
        default = defaults["A_matrix_regularization"] |> Float32

        "--plot_and_store_predictions"
        help = "Whether to plot and store predictions during training."
        arg_type = Bool
        default = defaults["plot_and_store_predictions"] |> Bool

        "--dt_switch"
        help = "Time delta for the open switchting time search interval (dt_switch, t_end)"
        arg_type = Float32
        default = get(defaults, "dt_switch", 1f-4) |> Float32

        "--dt_diag"
        help = "Time delta for the evaluation of the diagonal pattern."
        arg_type = Float32
        default = get(defaults, "dt_diag", 1f-4) |> Float32

        "--dz_tangent_threshold"
        help = "Threshold for the derivative of the solution to be considered zero."
        arg_type = Float32
        default = get(defaults, "dz_tangent_threshold", 1f-12) |> Float32

        "--max_iterations"
        help = "Maximum number of iterations for the root finding algorithm."
        arg_type = Int
        default = get(defaults, "max_iterations", 10000) |> Int

        "--max_region_width"
        help = "Maximum width of the region for the root finding algorithm."
        arg_type = Float32
        default = get(defaults, "max_region_width", 5.0f0) |> Float32

        "--epsilon_zero"
        help = "Tolerance interval width for the root finding algorithm."
        arg_type = Float32
        default = get(defaults, "epsilon_zero", 1f-8) |> Float32

        "--delta_zero"
        help = "Tolerance absolute value for the root finding algorithm."
        arg_type = Float32
        default = get(defaults, "delta_zero", 1f-8) |> Float32

        "--eigen_gap_threshold"
        help = "Threshold for the gap between eigenvalues to be considered zero."
        arg_type = Float32
        default = get(defaults, "eigen_gap_threshold", 1e-10) |> Float32

        "--degenerate_perturbation_strength"
        help = "Strength of the perturbation added to the matrix to avoid degeneracy."
        arg_type = Float32
        default = get(defaults, "degenerate_perturbation_strength", 1e-6) |> Float32


    end
    return settings
end

"""
    parse_commandline()

Parses all commandline arguments for execution of `main.jl`.
"""
parse_commandline() = parse_args(argtable())