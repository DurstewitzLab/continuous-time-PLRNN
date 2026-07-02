using LinearAlgebra
using Zygote
using ChainRulesCore
using Random
using Base.Threads

# Convert boolean pattern to bitmask (faster than tuple for vector conversion)
# Returns UInt32 for P <= 32, UInt64 for P <= 64
@inline function get_diagonal_pattern(z_current::AbstractVector, M::Int, P::Int)
    pattern = z_current[end-(P-1):end] .> 0
    if P <= 32
        bitmask = zero(UInt32)
        for i in 1:P
            if pattern[i]
                bitmask |= (one(UInt32) << (i - 1))
            end
        end
        return bitmask
    else
        bitmask = zero(UInt64)
        for i in 1:P
            if pattern[i]
                bitmask |= (one(UInt64) << (i - 1))
            end
        end
        return bitmask
    end
end

# Convert bitmask back to Float32 vector
@inline function bitmask_to_diag_vec(bitmask::Union{UInt32, UInt64}, M::Int, P::Int)
    diag_vec = Vector{Float32}(undef, M)
    diag_vec[1:(M-P)] .= 1.0f0
    for i in 1:P
        diag_vec[M-P+i] = ((bitmask >> (i - 1)) & 1) == 1 ? 1.0f0 : 0.0f0
    end
    return diag_vec
end

# Thread-safe per-batch cache
struct BatchCache{K,V}
    dict::Dict{K,V}
    lock::ReentrantLock
end

BatchCache{K,V}() where {K,V} = BatchCache(Dict{K,V}(), ReentrantLock())


# Custom LU solve for possible future differentiable cache
inv_lu(A_fact, A, x) = A_fact \ x



function compute_representation(A, W, h, diag_vec)
    Wsub   = A + W * Diagonal(diag_vec)
    λ, Pmat = eigen(Wsub, sortby = λ -> (imag(λ) != 0, real(λ)))
    P_fact  = lu(Pmat)
    h̃       = -(Wsub \ h)
    return (λ, Pmat, P_fact, h̃)
end


function get_cached_representation!(
        cache::BatchCache,
        A, W, h,
        diagonal_pattern,
        M::Int, P::Int)

    diag_vec = bitmask_to_diag_vec(diagonal_pattern, M, P)

    # Cache lookup (non-differentiable — returns plain values)
    local val
    found = false
    lock(cache.lock) do
        if haskey(cache.dict, diagonal_pattern)
            val   = cache.dict[diagonal_pattern]
            found = true
        end
    end

    if !found
        val = compute_representation(A, W, h, diag_vec)
        lock(cache.lock) do
            # double-checked: another thread may have stored it while we computed
            if !haskey(cache.dict, diagonal_pattern)
                cache.dict[diagonal_pattern] = val
            end
        end
    end

    return val
end

function SumRep_sort_fully_cached(
        cache::BatchCache,
        A,
        W,
        h,
        z_diag,
        z0,
        M::Int,
        P::Int)

    diagonal_pattern = get_diagonal_pattern(z_diag, M, P)

    λ, Pmat, P_fact, h̃ =
        get_cached_representation!(cache, A, W, h, diagonal_pattern, M, P)

    c   = inv_lu(P_fact, Pmat, z0 .- h̃)
    c̃   = Pmat .* transpose(c)

    return c̃, λ, h̃
end

