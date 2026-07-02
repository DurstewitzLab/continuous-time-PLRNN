module Models

using Lux, LinearAlgebra, Flux
using Lux: zeros32

using ..Utilities
using ..ContUtilities

export contALRNN,
    ALRNNHyperConfig,
    uniform_init,
    general_OHL_init

include("initialization.jl")
include("cont_alrnn.jl")
include("threadsmapdiff.jl")


end