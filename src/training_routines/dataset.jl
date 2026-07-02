using NPZ
using Flux: cpu

abstract type AbstractDataset end

struct Dataset{T, N, A <: AbstractArray{T, N}, v <: AbstractArray{T}} <: AbstractDataset
    X::A
    tvec::v
    name::String
end

function Dataset(path::String, time_path::String, name::String, Δt; device = cpu, dtype = Float32)
    X = npzread(path) .|> dtype |> device
    @assert ndims(X) ∈ (2, 3) "Data must be 2 or 3-dimensional but is $(ndims(X))-dimensional."
    
    # Convert 2D data to 3D with single trial
    if ndims(X) == 2
        X = reshape(X, size(X)..., 1)
    end
    
    T = size(X, 1)
    n_trials = size(X, 3)
    
    if time_path != ""
        tvec = npzread(time_path) .|> dtype |> device
        @assert ndims(tvec) ∈ (1, 2) "tvec must be 1 or 2-dimensional but is $(ndims(tvec))-dimensional."
        # Convert 1D tvec to 2D: replicate for all trials (shared time grid)
        if ndims(tvec) == 1
            tvec = repeat(reshape(tvec, length(tvec), 1), 1, n_trials)
        end
    else    
        tvec_single = collect(range(zero(dtype), T * dtype(Δt), length = T)) |> device
        tvec = repeat(reshape(tvec_single, T, 1), 1, n_trials)
    end
    
    @assert size(X, 1) == size(tvec, 1) "X and tvec must have the same time length"
    @assert size(X, 3) == size(tvec, 2) "X and tvec must have the same number of trials"
    
    return Dataset(X, tvec, name)
end

Dataset(path::String, time_path::String, Δt; device = cpu, dtype = Float32) =
    Dataset(path, time_path, "", Δt; device = device, dtype = dtype)

# Convenience constructor for backward compatibility (single trial, no time_path)
function load_dataset(path::String, time_path::String, Δt; device = cpu)
    return Dataset(path, time_path, Δt; device = device, dtype = Float32)
end


@inbounds """
    sample_sequence(dataset, sequence_length, trial_index)

Sample a sequence of length `T̃` from trial `j` of time series X.
"""
function sample_sequence(D::Dataset{T_, 3, A, v}, T̃::Int, j::Int) where {T_, A, v}
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return @views D.X[i:i+T̃, :, j], D.tvec[i:i+T̃, j]
end

"""
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement, sampling from all trials).
"""
function sample_batch(D::Dataset{T_, 3, A, v}, T̃::Int, S::Int) where {T_, A, v}
    _, N, n = size(D.X)
    Xs = similar(D.X, N, S, T̃ + 1)
    tvecs = similar(D.tvec, S, T̃ + 1)
    Threads.@threads for i = 1:S
        X, t = sample_sequence(D, T̃, rand(1:n))
        @views Xs[:, i, :] .= X'
        tvecs[i, :] .= t
    end
    return Xs, tvecs
end

