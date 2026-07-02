using LuxCore, Random
using Lux: zeros32

# Solver hyperparameters for `compute_alrnn_trajectory`. Static, non-trainable
# configuration that travels with the model (kept out of `ps`/`st`).
Base.@kwdef struct ALRNNHyperConfig
    max_iterations::Int           = 10000
    Δt_diag::Float32              = 1f-4
    Δt_switch::Float32            = 1f-4
    max_region_width::Float32     = Inf32
    dz_tangent_threshold::Float32 = 1f-12
    ϵ_zero::Float32               = 1f-8
    δ_zero::Float32               = 1f-8
    λ_gap_threshold::Float32      = 1e-10
    dW_strength::Float32          = 1e-6
end

# Build the hyperparameter config from parsed command line arguments.
# Missing keys fall back to struct defaults (e.g. old saved args.json files).
function ALRNNHyperConfig(args::AbstractDict)
    defaults = ALRNNHyperConfig()
    return ALRNNHyperConfig(;
        max_iterations       = get(args, "max_iterations", defaults.max_iterations),
        Δt_diag              = get(args, "dt_diag", defaults.Δt_diag),
        Δt_switch            = get(args, "dt_switch", defaults.Δt_switch),
        max_region_width     = get(args, "max_region_width", defaults.max_region_width),
        dz_tangent_threshold = get(args, "dz_tangent_threshold", defaults.dz_tangent_threshold),
        ϵ_zero               = get(args, "epsilon_zero", defaults.ϵ_zero),
        δ_zero               = get(args, "delta_zero", defaults.δ_zero),
        λ_gap_threshold      = get(args, "eigen_gap_threshold", defaults.λ_gap_threshold),
        dW_strength          = get(args, "degenerate_perturbation_strength", defaults.dW_strength),
    )
end

struct contALRNN{A0, W0, h0} <: LuxCore.AbstractLuxLayer
    M::Int #dimensionality of the PLRNN
    P::Int #number of piecewise linear regions
    A_init::A0 #linear dynamics matrix creation function
    W_init::W0 #nonlinear dynamics matrix creation function
    h_init::h0 #bias vector creation function
    hyper_config::ALRNNHyperConfig
end

function Base.show(io::IO, d::contALRNN)
    print(io, "contALRNN($(d.M), $(d.P)")
    return print(io, ")")
end

function contALRNN(M::Int, P::Int; A_init = glorot_uniform, W_init = glorot_uniform, h_init = zeros32, hyper_config = ALRNNHyperConfig())
    return contALRNN{typeof(A_init), typeof(W_init), typeof(h_init)}(M, P, A_init, W_init, h_init, hyper_config)
end


function LuxCore.initialparameters(rng::AbstractRNG, model::contALRNN)
    return (A = log.(diag(normalized_positive_definite(rng,model.M))), 
            W = gaussian_init(rng, model.M, model.M), 
            h = gaussian_init(rng, model.M, 1)[:])
end



LuxCore.initialstates(::AbstractRNG, ::contALRNN) = NamedTuple()
LuxCore.parameterlength(model::contALRNN) = model.M + model.M^2 + model.M
LuxCore.statelength(model::contALRNN) = 0

outputsize(model::contALRNN, _, ::AbstractRNG) = (model.M,)

###Taking the idea from https://lux.csail.mit.edu/stable/manual/dispatch_custom_input as custom input type
function (model::contALRNN)((z₁, t_points)::Tuple{AbstractVecOrMat, AbstractVector}, ps, st::NamedTuple) #z::AbstractMatrix
    hyper_config = model.hyper_config
    z_points = compute_alrnn_trajectory(t_points, ps.A, ps.W, ps.h, z₁, model.P;
    max_iterations=hyper_config.max_iterations,
    Δt_diag=hyper_config.Δt_diag,
    Δt_switch=hyper_config.Δt_switch,
    max_region_width=hyper_config.max_region_width,
    dz_tangent_threshold=hyper_config.dz_tangent_threshold,
    ϵ_zero=hyper_config.ϵ_zero,
    δ_zero=hyper_config.δ_zero)
    return z_points, st
end

function (model::contALRNN)((z₁, t_points)::Tuple{AbstractMatrix, AbstractMatrix}, ps, st::NamedTuple)
    batch_size = size(z₁, 2)
    T = size(t_points, 2)
    N = size(z₁, 1)

    X̃ = ThreadsX.map(1:batch_size) do i 
        Lux.apply(model, (@view(z₁[:, i]), @view(t_points[i, :])), ps, st)[1] 
    end
    return permutedims(cat(X̃..., dims=3), (1, 3, 2)), st
end

