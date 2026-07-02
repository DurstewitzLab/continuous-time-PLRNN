using LuxCore, Random, LinearAlgebra, FillArrays
using Lux: zeros32, IntegerType, StaticBool, BoolType, True, static, False, has_bias
using ConcreteStructs: @concrete

abstract type AffineObservationModel <: ObservationModel end

#"https://lux.csail.mit.edu/stable/manual/migrate_from_flux#migrate-from-flux"


#just copy the implementation of struct Dense

@concrete struct Identity <: AffineObservationModel #AbstractLuxLayer #@concrete but I do not know which package it is from
    N <: IntegerType
    M <: IntegerType
    B_init
    b_init
    L_init
    use_bias <: StaticBool
end

function Base.show(io::IO, O::Identity)
    print(io, "Identity($(O.M) => $(O.N)")
    has_bias(O) || print(io, ", use_bias=false")
    return print(io, ")")
end


function Identity(mapping::Pair{<:IntegerType, <:IntegerType}; kwargs...)
    return Identity(first(mapping), last(mapping); kwargs...)
end

function Identity(M::IntegerType, N::IntegerType;
        B_init=nothing, b_init=nothing, L_init=nothing, use_bias::BoolType=False())
    return Identity(M, N, B_init, b_init, L_init, static(use_bias))
end


# `L` is a parameter, as well as the bias b if use_bias=true
function LuxCore.initialparameters(rng::AbstractRNG, O::Identity)
    L = if O.L_init === nothing
        L = initialize_L(O.M, O.N)
    else
        O.L_init(rng, O.M, O.N) #L_init and so are the initializer functions for the actual weight?
    end
    has_bias(O) || return (; L = L)
    return (; L = L, b=zeros32(O.N))
end

LuxCore.parameterlength(O::Identity) = (O.M===O.N) ? 0 + has_bias(O)*O.N : (O.M - O.N)*O.N + has_bias(O)*O.N

# `B` is a state
function LuxCore.initialstates(::AbstractRNG, O::Identity) 
    return (B = [I(O.N) zeros(Float32, O.N, O.M - O.N)], ) #comma has to be there for it to be a named tuple
end

LuxCore.statelength(O::Identity) = O.M*O.N

init_state(O::Identity, x::AbstractVecOrMat, ps, st::NamedTuple) =
    if ps.L === nothing
        return x
    else
        return [x; ps.L * x]
    end


(O::Identity)(z::AbstractArray, ps, st::NamedTuple) = forward(O, z, ps, st::NamedTuple)

@inbounds forward(O::Identity, z::AbstractVector, ps, st::NamedTuple; return_view::Bool = false) =
    return_view ? @view(z[1:O.N]) : z[1:O.N]

@inbounds forward(O::Identity, z::AbstractMatrix, ps, st::NamedTuple; return_view::Bool = false) =
    return_view ? @view(z[1:O.N, :]) : z[1:O.N, :]

@inbounds forward(O::Identity, z::AbstractArray{T, 3}, ps, st::NamedTuple; return_view::Bool = false,
) where {T} = return_view ? @view(z[1:O.N, :, :]) : z[1:O.N, :, :]


apply_inverse(O::Identity, x::AbstractVector, ps, st::NamedTuple) =
    [x; Zeros{eltype(x)}(O.M - O.N)]

apply_inverse(O::Identity, x::AbstractMatrix, ps, st::NamedTuple) =
    [x; Zeros{eltype(x)}(O.M - O.N, size(x, 2))]

apply_inverse(O::Identity, x::AbstractArray{T, 3}, ps, st::NamedTuple) where {T} =
    [x; Zeros{eltype(x)}(O.M - O.N, size(x, 2), size(x, 3))]

inverse(O::Identity) = O.B'


# Initialize L matrix
function initialize_L(M::Int, N::Int)
    if M == N
        return nothing
    else
        return glorot_uniform(M - N, N)
    end
end

