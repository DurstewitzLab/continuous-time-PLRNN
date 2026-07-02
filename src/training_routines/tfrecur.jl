using ..Models
using ..Utilities
using ..ObservationModels

using Random: AbstractRNG


using LuxCore
using LuxCore: initialstates

abstract type AbstractTFRecur <: LuxCore.AbstractLuxContainerLayer{(:cont_model, :obs_model)} end
(tfrec::AbstractTFRecur)((X, t_points)::Tuple{AbstractArray{T, 3}, AbstractMatrix}, ps, st) where {T} = forward(tfrec, (X, t_points), ps, st)


"""
    forward(tfrec, X)

Forward pass using teacher forcing. If the latent dimension of
the RNN is larger than the dimension the observations live in, 
partial teacher forcing of the first `N = size(X, 1)` neurons is
used. Initializing latent state `z₁` is taken care of by the observation model.
"""

function forward(tfrec::AbstractTFRecur, (X, t_points)::Tuple{AbstractArray{T, 3}, AbstractMatrix}, ps, st::NamedTuple) where {T}
    N, S, T̃ = size(X)
    M = tfrec.cont_model.M

    τ = tfrec.τ

    # number of forced states
    D = min(N, M)

    # precompute forcing signals
    Z⃰= apply_inverse(tfrec.obs_model, X, ps.obs_model, st.obs_model)

    z0 = @views init_state(tfrec.obs_model, X[:, :, 1], ps.obs_model, st.obs_model)

    # initialize latent state
    st = Lux.update_state(st, :z0, z0)

    Z = reshape(Float32[], M, S, 0)

    for t in 1:τ:T̃
        t̃ = min(t + τ, T̃) #because tilde T might not be a multiple of tau
        z, st_ = tfrec((Z⃰[1:D, :, t̃], t_points[:,t:t̃]), ps, st)
        Z = cat(Z, z, dims = 3)
        st = st_
    end

    # reshape to 3d array and return
    return Z, st
end

"""
Inspired by `Flux.Recur` struct, which by default has no way
of incorporating teacher forcing.

This is just a convenience wrapper around stateful models,
to be used during training.
"""
mutable struct TFRecur{M, O <: ObservationModel} <: AbstractTFRecur
    cont_model::M
    obs_model::O
    τ::Int
end

# Add initialstates function to properly initialize the state
function LuxCore.initialstates(rng::AbstractRNG, tfrec::TFRecur)
    return (;
        cont_model=initialstates(rng, tfrec.cont_model),
        obs_model=initialstates(rng, tfrec.obs_model),
        z0=nothing  # Will be initialized on first forward pass
    )
end

function (tfrec::TFRecur)((x, t_points)::Tuple{AbstractMatrix, AbstractMatrix}, ps, st::NamedTuple)
    z0 = st.z0

    z, _ = tfrec.cont_model((z0, t_points), ps.cont_model, st.cont_model)

    z0 = z[:, :, end]

    z0 = force(z0, x)

    st = Lux.update_state(st, :z0, z0)

    return z[:, :, 2:end], st
end
