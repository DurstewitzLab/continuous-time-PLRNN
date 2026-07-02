module ContUtilities

using LinearAlgebra

include("cont_utils_functions.jl")
export SumRep_sort,
    ContPLRNNSolution,
    ContPLRNNSolutionDerivative

include("utils_branch_and_prune.jl")
export roots_multiple

include("cont_caches_preliminary.jl")
export BatchCache
SumRep_sort_fully_cached


include("cont_utils_solution.jl")
export compute_alrnn_trajectory,
compute_alrnn_trajectory_fully_cached



end