using Random

function normalized_positive_definite(rng::AbstractRNG, M::Int)
    R = randn(rng, Float32, M, M)
    K = R'R ./ M + I
    λ = maximum(abs.(eigvals(K)))
    return K ./ λ
end

function uniform_init(rng::AbstractRNG, shape::Tuple; eltype::Type{T} = Float32) where {T <: AbstractFloat}
    @assert length(shape) < 3
    din = Float32(shape[end])
    r = 1 / √din
    return rand(rng, -r, r, shape...)
end

function gaussian_init(rng::AbstractRNG, M::Int, N::Int)
    return Float32.(randn(rng, Float32, M, N) .* 0.01)
end

function initialize_A_W_h(rng::AbstractRNG, M::Int)
    A = diag(normalized_positive_definite(rng, M))
    W = gaussian_init(rng, M, M)
    h = zeros(Float32, M)
    return A, W, h
end

function initialize_L(rng::AbstractRNG, M::Int, N::Int)
    if M == N
        L = nothing
    else
        L = uniform_init(rng, (M - N, N))
    end
    return L
end