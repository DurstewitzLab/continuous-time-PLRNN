using LinearAlgebra
using Random

using Zygote

Zygote.@nograd function min_gap(λ)
    # Since eigenvalues are already sorted, we only need to check adjacent pairs
    # This reduces complexity from O(n²) to O(n)
    if length(λ) < 2
        return Inf
    end
    minimum(abs(λ[i] - λ[i+1]) for i in 1:(length(λ)-1))
end

# Helper function to compute perturbed decomposition without gradients
Zygote.@nograd function compute_perturbed_decomposition(W, h; dW_strength::Float32 = Float32(1e-6))
    W_perturbed = W + Float32.(randn(size(W))*dW_strength)
    λ, P = eigen(W_perturbed, sortby = λ -> (imag(λ) != 0, real(λ)))
    h̃ = -(W \ h)
    return λ, P, h̃
end

"""
Sum representation of the system sorted by real and complex eigenvalues and then by the real part of the eigenvalues
"""
function SumRep_sort(W::AbstractMatrix, z0::AbstractVector, h::AbstractVector
    ;λ_gap_threshold::Float32 = Float32(1e-10), dW_strength::Float32 = Float32(1e-6))
    λ, P = eigen(W, sortby = λ -> (imag(λ) != 0, real(λ)))
    # Check if perturbation is needed (use @nograd to prevent gradient flow through branch decision)
    needs_perturbation = (min_gap(λ) < λ_gap_threshold)
    if needs_perturbation
        # Prevent gradient flow to A and W when matrix is degenerate
        # Use perturbed matrix consistently for both eigen decomposition and h̃ computation
        λ, P, h̃ = compute_perturbed_decomposition(W, h; dW_strength=dW_strength)
    else
        h̃ = -(W \ h)
    end
    c = P \ (z0 .- h̃)
    c̃ = P .* transpose(c)
    return c̃, λ, h̃
end


@inline function exp_outer(λ::AbstractVector, tvec::AbstractVector)
    # Create K x N matrix where K = length(λ), N = length(tvec)
    λ_matrix = reshape(λ, :, 1)  # K x 1
    t_matrix = reshape(tvec, 1, :)  # 1 x N
    @. exp(λ_matrix * t_matrix)  # K x N matrix
end

function ContPLRNNSolution(tvec::AbstractVector, c̃::AbstractMatrix, λ::AbstractVector, h̃::AbstractVector)
    expmat = exp_outer(λ, tvec)  # K x N
    c̃ * expmat .+ h̃             # M x N
end

# Continuous solution for single neuron
ContPLRNNSolution(i::Integer, t::Union{Real,Complex}, c̃::AbstractMatrix, λ::AbstractVector, h̃::AbstractVector) =
    sum(c̃[i,:] .* exp.(λ .* t)) + h̃[i]

# Derivative for single neuron
ContPLRNNSolutionDerivative(i::Integer, t::Union{Real,Complex}, c̃::AbstractMatrix, λ::AbstractVector) =
    sum(λ .* c̃[i,:] .* exp.(λ .* t))

function ContPLRNNSolution(t::T, c̃::AbstractMatrix{<:Union{T, Complex{T}}}, λ::AbstractVector{<:Union{T, Complex{T}}}, h̃::AbstractVector{<:T}) where T <: AbstractFloat
    return real.(c̃ * exp.(λ * t)) .+ h̃
end
