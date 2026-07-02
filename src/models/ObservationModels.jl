module ObservationModels

using Lux

using ..Utilities

abstract type ObservationModel <: AbstractLuxLayer end
(O::ObservationModel)(z::AbstractArray, ps, st::NamedTuple) = forward(O, z, ps, st)

export ObservationModel, Identity, apply_inverse, init_state

include("affine.jl")

end