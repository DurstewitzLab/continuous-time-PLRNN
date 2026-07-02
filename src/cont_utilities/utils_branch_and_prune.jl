using LinearAlgebra
using Zygote
using Roots: find_zero
using BranchAndPrune: DepthFirst, BranchAndPruneSearch, BPNode, prune!#, SearchState

"""
Structures defined to hold the information needed for the BranchAndPrune Search.
Modelled after Root (IntervalRootFinding.jl) and SearchState (BranchAndPrune.jl)
"""

struct RootMultiple{T<:AbstractFloat}
    region::Tuple{T,T}
    id_active::Vector{Int}
end

struct SearchState_Root{S,REGION,T}
    search_order::S
    tree::BPNode{REGION}
    iteration::Int
    t_root_min::T
    t_root_min_index::Int
end

function SearchState_Root(S, initial_region::REGION) where REGION
    root = BPNode(:working, initial_region, nothing, :left)
    return SearchState_Root(S(root), root, 1, Inf32, 0)
end

"""
Wrapper around BranchAndPruneSearch to allow defining a custom iterate method
without overwriting the method on the external BranchAndPrune type (method piracy).
"""
struct RootMultipleSearch{S,REGION,F,G}
    search::BranchAndPruneSearch{S,REGION,F,G}
end

function RootMultipleSearch(S, process::F, bisect::G, initial_region::REGION; kwargs...) where {F,G,REGION}
    return RootMultipleSearch{S,REGION,F,G}(BranchAndPruneSearch{S,REGION,F,G}(process, bisect, initial_region))
end

"""
Parameter decomposition of the system into real and complex eigenvalues and coefficients
"""
function Parameter_Decomposition(λ::AbstractVector{<:Union{Complex{T},T}}, c̃::AbstractMatrix{<:Union{Complex{T},T}}, id_complex::Int) where T<:AbstractFloat
    n_real = id_complex - 1
    n_complex = length(λ) - id_complex + 1
    n_rows = size(c̃, 1)
    n_complex_pairs = n_complex ÷ 2

    # Pre-allocate result arrays
    λ_real = Vector{T}(undef, n_real)
    c̃_real = Matrix{T}(undef, n_rows, n_real)
    a_vec = Vector{T}(undef, n_complex_pairs)
    b_vec = Vector{T}(undef, n_complex_pairs)
    v_vec = Matrix{T}(undef, n_rows, n_complex_pairs)
    w_vec = Matrix{T}(undef, n_rows, n_complex_pairs)

    # Extract real eigenvalues and coefficients
    @inbounds @simd for i in 1:n_real
        λ_real[i] = real(λ[i])
    end

    @inbounds for j in 1:n_real
        @simd for i in 1:n_rows
            c̃_real[i, j] = real(c̃[i, j])
        end
    end

    # Extract complex eigenvalues: a_vec and b_vec from λ_complex[1:2:end]
    # λ_complex = λ[id_complex:end], so λ_complex[1:2:end] maps to λ[id_complex], λ[id_complex+2], ...
    @inbounds @simd for k in 1:n_complex_pairs
        idx = id_complex + 2 * (k - 1)  # Maps to λ_complex[1:2:end] indices
        λ_val = λ[idx]
        a_vec[k] = real(λ_val)
        b_vec[k] = imag(λ_val)
    end

    # Extract complex coefficients: v_vec and w_vec from c̃_complex[:,1:2:end]
    @inbounds for k in 1:n_complex_pairs
        col_idx = id_complex + 2 * (k - 1)  # Maps to c̃_complex[:,1:2:end] column indices
        @simd for i in 1:n_rows
            c_val = c̃[i, col_idx]
            v_vec[i, k] = real(c_val)
            w_vec[i, k] = imag(c_val)
        end
    end

    return λ_real, c̃_real, a_vec, b_vec, v_vec, w_vec
end


"""
Function calculating the interval bounds of the real part of the system and the complex part of the system,
i.e. needing sine and cosine expressions
For c * exp(λ * t) where t ∈ [t_inf, t_sup]:
- If λ > 0: exp is increasing → exp(λ * t_inf) < exp(λ * t_sup)
- If λ < 0: exp is decreasing → exp(λ * t_inf) > exp(λ * t_sup)
- If λ = 0: exp is constant → exp(0) = 1
"""
function exp_real_interval(c̃_real::AbstractMatrix{<:T}, λ_real::AbstractVector{<:T}, t_inf::T, t_sup::T) where T<:AbstractFloat
    n_rows, n_cols = size(c̃_real)

    # Pre-compute exponential values once (needed for all rows)
    exp_λ_t_inf = Vector{T}(undef, n_cols)
    exp_λ_t_sup = Vector{T}(undef, n_cols)
    @inbounds @simd for j in 1:n_cols
        exp_λ_t_inf[j] = exp(λ_real[j] * t_inf)
        exp_λ_t_sup[j] = exp(λ_real[j] * t_sup)
    end

    # Pre-allocate result vectors
    lower_bound = zeros(T, n_rows)
    upper_bound = zeros(T, n_rows)

    # Fused loop: iterate over rows and columns, compute both bounds in one pass
    @inbounds for i in 1:n_rows
        @simd for j in 1:n_cols
            c_val = c̃_real[i, j]
            exp_inf_val = exp_λ_t_inf[j]
            exp_sup_val = exp_λ_t_sup[j]

            # Determine which exponential value is min and which is max
            # (depends on sign of λ: if λ > 0, exp_inf < exp_sup; if λ < 0, exp_inf > exp_sup)
            exp_min = min(exp_inf_val, exp_sup_val)
            exp_max = max(exp_inf_val, exp_sup_val)

            if c_val < 0
                # When c_val < 0: multiplying by negative coefficient reverses min/max
                # Lower bound uses exp_max (since c < 0, larger exp gives smaller product)
                # Upper bound uses exp_min (since c < 0, smaller exp gives larger product)
                lower_bound[i] += c_val * exp_max
                upper_bound[i] += c_val * exp_min
            else
                # When c_val >= 0: normal ordering
                # Lower bound uses exp_min, upper bound uses exp_max
                lower_bound[i] += c_val * exp_min
                upper_bound[i] += c_val * exp_max
            end
        end
    end

    return lower_bound, upper_bound
end

function quadrant(x::T) where T<:AbstractFloat
    const_halfpi = T(π / 2)  # Compile-time constant
    return floor(Int, mod(x / const_halfpi, 4))
end

function sin_interval_vec(lo_vec::AbstractVector{<:T}, hi_vec::AbstractVector{<:T}) where T<:AbstractFloat
    n = length(lo_vec)
    length(hi_vec) != n && throw(DimensionMismatch("lo_vec and hi_vec must have same length"))

    # Pre-allocate result matrix (2 rows: [lower, upper])
    result = Matrix{T}(undef, 2, n)

    const_PI_HI = T(π)
    const_2PI_HI = T(2π)
    const_one = one(T)
    const_neg_one = -one(T)

    @inbounds for i in 1:n
        lo = lo_vec[i]
        hi = hi_vec[i]
        d = hi - lo

        if d ≥ const_2PI_HI
            result[1, i] = const_neg_one
            result[2, i] = const_one
        else
            lo_quadrant = quadrant(lo)
            hi_quadrant = quadrant(hi)

            if lo_quadrant == hi_quadrant
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                elseif (lo_quadrant == 1) || (lo_quadrant == 2)
                    # decreasing
                    result[1, i] = sin(hi)
                    result[2, i] = sin(lo)
                else
                    # increasing
                    result[1, i] = sin(lo)
                    result[2, i] = sin(hi)
                end
            elseif lo_quadrant == 3 && hi_quadrant == 0
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                else
                    result[1, i] = sin(lo)
                    result[2, i] = sin(hi)
                end
            elseif lo_quadrant == 1 && hi_quadrant == 2
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                else
                    result[1, i] = sin(hi)
                    result[2, i] = sin(lo)
                end
            elseif (lo_quadrant == 0 || lo_quadrant == 3) && (hi_quadrant == 1 || hi_quadrant == 2)
                sin_lo = sin(lo)
                sin_hi = sin(hi)
                result[1, i] = min(sin_lo, sin_hi)
                result[2, i] = const_one
            elseif (lo_quadrant == 1 || lo_quadrant == 2) && (hi_quadrant == 3 || hi_quadrant == 0)
                sin_lo = sin(lo)
                sin_hi = sin(hi)
                result[1, i] = const_neg_one
                result[2, i] = max(sin_lo, sin_hi)
            else
                # (lo_quadrant == 0 && hi_quadrant == 3) || (lo_quadrant == 2 && hi_quadrant == 1)
                result[1, i] = const_neg_one
                result[2, i] = const_one
            end
        end
    end

    return result
end

function cos_interval_vec(lo_vec::AbstractVector{<:T}, hi_vec::AbstractVector{<:T}) where T<:AbstractFloat
    n = length(lo_vec)
    length(hi_vec) != n && throw(DimensionMismatch("lo_vec and hi_vec must have same length"))

    # Pre-allocate result matrix (2 rows: [lower, upper])
    result = Matrix{T}(undef, 2, n)

    const_PI_HI = T(π)
    const_2PI_HI = T(2π)
    const_one = one(T)
    const_neg_one = -one(T)

    @inbounds for i in 1:n
        lo = lo_vec[i]
        hi = hi_vec[i]
        d = hi - lo

        if d ≥ const_2PI_HI
            result[1, i] = const_neg_one
            result[2, i] = const_one
        else
            lo_quadrant = quadrant(lo)
            hi_quadrant = quadrant(hi)

            if lo_quadrant == hi_quadrant
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                elseif (lo_quadrant == 2) || (lo_quadrant == 3)
                    # increasing
                    result[1, i] = cos(lo)
                    result[2, i] = cos(hi)
                else
                    # decreasing
                    result[1, i] = cos(hi)
                    result[2, i] = cos(lo)
                end
            elseif lo_quadrant == 2 && hi_quadrant == 3
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                else
                    result[1, i] = cos(lo)
                    result[2, i] = cos(hi)
                end
            elseif lo_quadrant == 0 && hi_quadrant == 1
                if d ≥ const_PI_HI
                    result[1, i] = const_neg_one
                    result[2, i] = const_one
                else
                    result[1, i] = cos(hi)
                    result[2, i] = cos(lo)
                end
            elseif (lo_quadrant == 2 || lo_quadrant == 3) && (hi_quadrant == 0 || hi_quadrant == 1)
                cos_lo = cos(lo)
                cos_hi = cos(hi)
                result[1, i] = min(cos_lo, cos_hi)
                result[2, i] = const_one
            elseif (lo_quadrant == 0 || lo_quadrant == 1) && (hi_quadrant == 2 || hi_quadrant == 3)
                cos_lo = cos(lo)
                cos_hi = cos(hi)
                result[1, i] = const_neg_one
                result[2, i] = max(cos_lo, cos_hi)
            else
                # (lo_quadrant == 3 && hi_quadrant == 2) || (lo_quadrant == 1 && hi_quadrant == 0)
                result[1, i] = const_neg_one
                result[2, i] = const_one
            end
        end
    end

    return result
end

function exp_complex_interval(
    a_vec::AbstractVector{<:T},
    b_vec::AbstractVector{<:T},
    v_vec::AbstractMatrix{<:T},
    w_vec::AbstractMatrix{<:T},
    t_inf::T,
    t_sup::T) where {T<:AbstractFloat}

    n_complex = length(a_vec)
    n_rows = size(v_vec, 1)

    # Use optimized interval functions
    cos_b = cos_interval_vec(b_vec * t_inf, b_vec * t_sup)  # Returns 2×n_complex matrix
    sin_b = sin_interval_vec(b_vec * t_inf, b_vec * t_sup)   # Returns 2×n_complex matrix

    # Pre-compute exponential values
    exp_a_inf = Vector{T}(undef, n_complex)
    exp_a_sup = Vector{T}(undef, n_complex)
    @inbounds @simd for i in 1:n_complex
        exp_a_inf[i] = exp(a_vec[i] * t_inf)
        exp_a_sup[i] = exp(a_vec[i] * t_sup)
    end

    # Pre-allocate arrays for exp*cos and exp*sin bounds
    exp_cos_inf = Vector{T}(undef, n_complex)
    exp_cos_sup = Vector{T}(undef, n_complex)
    exp_sin_inf = Vector{T}(undef, n_complex)
    exp_sin_sup = Vector{T}(undef, n_complex)

    # Compute min/max of all 4 products for each complex mode
    @inbounds @simd for i in 1:n_complex
        # For cos: compute all 4 products and take min/max
        prod1 = exp_a_inf[i] * cos_b[1, i]  # exp_a_inf * cos_low
        prod2 = exp_a_inf[i] * cos_b[2, i]  # exp_a_inf * cos_high
        prod3 = exp_a_sup[i] * cos_b[1, i]  # exp_a_sup * cos_low
        prod4 = exp_a_sup[i] * cos_b[2, i]  # exp_a_sup * cos_high
        exp_cos_inf[i] = min(prod1, prod2, prod3, prod4)
        exp_cos_sup[i] = max(prod1, prod2, prod3, prod4)

        # For sin: compute all 4 products and take min/max
        prod1_sin = exp_a_inf[i] * sin_b[1, i]  # exp_a_inf * sin_low
        prod2_sin = exp_a_inf[i] * sin_b[2, i]  # exp_a_inf * sin_high
        prod3_sin = exp_a_sup[i] * sin_b[1, i]  # exp_a_sup * sin_low
        prod4_sin = exp_a_sup[i] * sin_b[2, i]  # exp_a_sup * sin_high
        exp_sin_inf[i] = min(prod1_sin, prod2_sin, prod3_sin, prod4_sin)
        exp_sin_sup[i] = max(prod1_sin, prod2_sin, prod3_sin, prod4_sin)
    end

    # Pre-allocate result vectors
    lower_bound_phase_v = zeros(T, n_rows)
    upper_bound_phase_v = zeros(T, n_rows)
    lower_bound_phase_w = zeros(T, n_rows)
    upper_bound_phase_w = zeros(T, n_rows)

    # Multiply by v_vec and accumulate
    @inbounds for i in 1:n_rows
        @simd for j in 1:n_complex
            v_val = v_vec[i, j]
            if v_val >= 0
                # v >= 0: lower = v*exp_cos_inf, upper = v*exp_cos_sup
                lower_bound_phase_v[i] += v_val * exp_cos_inf[j]
                upper_bound_phase_v[i] += v_val * exp_cos_sup[j]
            else
                # v < 0: lower = v*exp_cos_sup, upper = v*exp_cos_inf
                lower_bound_phase_v[i] += v_val * exp_cos_sup[j]
                upper_bound_phase_v[i] += v_val * exp_cos_inf[j]
            end
        end
    end

    # Multiply by -w_vec and accumulate
    @inbounds for i in 1:n_rows
        @simd for j in 1:n_complex
            w_val = w_vec[i, j]
            if w_val >= 0
                # w >= 0: -w * [sin_inf, sin_sup] = [-w*sin_sup, -w*sin_inf]
                lower_bound_phase_w[i] -= w_val * exp_sin_sup[j]
                upper_bound_phase_w[i] -= w_val * exp_sin_inf[j]
            else
                # w < 0: -w * [sin_inf, sin_sup] = [-w*sin_inf, -w*sin_sup]
                lower_bound_phase_w[i] -= w_val * exp_sin_inf[j]
                upper_bound_phase_w[i] -= w_val * exp_sin_sup[j]
            end
        end
    end

    # Combine sin and cos bounds
    lower_bound_total = 2 * (lower_bound_phase_v .+ lower_bound_phase_w)
    upper_bound_total = 2 * (upper_bound_phase_v .+ upper_bound_phase_w)

    return lower_bound_total, upper_bound_total
end


"""
Function calculating the interval bounds of the solution of the system, full expression
"""

function ContPLRNNSolution_interval(c̃::AbstractMatrix{<:Union{Complex{T},T}}, λ::AbstractVector{<:Union{Complex{T},T}}, h̃::AbstractVector{<:T}, t_inf::T, t_sup::T) where {T<:AbstractFloat}
    #split between real and complex parts
    id_complex = findfirst(!isreal, λ)

    #only real eigenvalues
    if isnothing(id_complex)
        lower_bound_real, upper_bound_real = exp_real_interval(real.(c̃), real.(λ), t_inf, t_sup)
        return lower_bound_real + h̃, upper_bound_real + h̃
    end

    #only complex eigenvalues
    if id_complex == 1
        _, _, a_vec, b_vec, v_vec, w_vec = Parameter_Decomposition(λ, c̃, id_complex)
        lower_bound_complex, upper_bound_complex = exp_complex_interval(a_vec, b_vec, v_vec, w_vec, t_inf, t_sup)
        return lower_bound_complex + h̃, upper_bound_complex + h̃
    end

    # # Use sentinel value when all eigenvalues are real
    # id_complex_for_decomp = isnothing(id_complex) ? length(λ) + 1 : id_complex

    # Use optimized parameter decomposition for all cases
    λ_real, c̃_real, a_vec, b_vec, v_vec, w_vec = Parameter_Decomposition(λ, c̃, id_complex)

    # Compute real part bounds (empty if no real eigenvalues)
    lower_bound_real, upper_bound_real = exp_real_interval(c̃_real, λ_real, t_inf, t_sup)

    # Compute complex part bounds (empty if no complex eigenvalues)
    lower_bound_complex, upper_bound_complex = exp_complex_interval(a_vec, b_vec, v_vec, w_vec, t_inf, t_sup)

    return lower_bound_real + lower_bound_complex + h̃, upper_bound_real + upper_bound_complex + h̃
end

""" 
Interval operations needed for the Newton Step on the Interval
"""
function contains_zero(lower_bound::AbstractVector{<:T}, upper_bound::AbstractVector{<:T}) where T<:AbstractFloat
    n = length(lower_bound)
    length(upper_bound) != n && throw(DimensionMismatch("lower_bound and upper_bound must have same length"))

    result = BitVector(undef, n)
    @inbounds @simd for i in 1:n
        result[i] = lower_bound[i] <= zero(T) <= upper_bound[i]
    end
    return result
end

function intersect_interval(lower_bound1, upper_bound1, lower_bound2, upper_bound2)
    return max.(lower_bound1, lower_bound2), min.(upper_bound1, upper_bound2)
end

function isempty_interval(lower_bound, upper_bound)
    return lower_bound .> upper_bound
end

function isstrictsubset_interval(lower_bound1, upper_bound1, lower_bound2, upper_bound2)
    return lower_bound2 .< lower_bound1 .&& upper_bound2 .> upper_bound1
end

"""
Bisection for the BranchAndPrune Search
"""
function bisect_region_multiple(root_multiple::RootMultiple)
    Y1, Y2 = bisect_interval(root_multiple.region)
    return RootMultiple(Y1, root_multiple.id_active), RootMultiple(Y2, root_multiple.id_active)
end


#could be rewritten to do correct rounding of m, depending on if it is lower or upper bound
function bisect_interval((a, b))
    m = (a + b) / 2  # The midpoint
    return (a, m), (m, b)
end

"""
Create small left node and rest in right. I am bad with names
"""
function bisect_region_left_multiple(root_multiple::RootMultiple, max_region_width::T) where {T<:AbstractFloat}
    Y1, Y2 = bisect_interval_left(root_multiple.region, max_region_width)
    return RootMultiple(Y1, root_multiple.id_active), RootMultiple(Y2, root_multiple.id_active)
end


#could be rewritten to do correct rounding of m, depending on if it is lower or upper bound
function bisect_interval_left((a, b), max_region_width::T) where {T<:AbstractFloat}
    if b - a > 2 * max_region_width #more than double the size of the max_interval => left interval
        return (a, a + max_region_width), (a + max_region_width, b)
    elseif b - a > max_region_width #more than the size of the interval => normal bisection
        return bisect_interval((a, b))
    end
    return (a, b)
end




""" Newton contraction step """

function contract_newton(
    c̃::AbstractMatrix{<:Union{Complex{T},T}},
    λ::AbstractVector{<:Union{Complex{T},T}},
    h̃::AbstractVector{<:T},
    t_inf::T,
    t_sup::T
) where {T<:AbstractFloat}
    # Compute derivative coefficients: dc̃ = c̃ * λ (element-wise)
    dc̃ = c̃ .* Transpose(λ)

    # Compute derivative bounds over the interval [t_inf, t_sup]
    # Use zero vector for h̃ since derivative doesn't depend on it
    df_lower, df_upper = ContPLRNNSolution_interval(dc̃, λ, zeros(T, length(h̃)), t_inf, t_sup)

    # Check if derivative contains zero (indicates potential issues)
    df_contains_zero = contains_zero(df_lower, df_upper)

    # Compute midpoint and function value at midpoint
    t_mid = (t_sup + t_inf) / 2
    f_mid = ContPLRNNSolution(t_mid, c̃, λ, h̃)

    # Pre-allocate result vectors (fused: compute bounds directly)
    n = length(f_mid)
    lower_bound = Vector{T}(undef, n)
    upper_bound = Vector{T}(undef, n)

    # Compute Newton steps for all elements (SIMD-friendly)
    # Process non-zero cases first for better SIMD performance
    @inbounds @simd for i in 1:n
        if !df_contains_zero[i]
            # Compute f(mid) / df for both bounds
            step_lower = f_mid[i] / df_lower[i]
            step_upper = f_mid[i] / df_upper[i]
            # Fuse: compute bounds directly (new_bound = mid - f(mid)/f'(mid))
            lower_bound[i] = t_mid - max(step_lower, step_upper)
            upper_bound[i] = t_mid - min(step_lower, step_upper)
        end
    end

    # Fix zero derivative cases (separate pass, typically few elements)
    @inbounds for i in 1:n
        if df_contains_zero[i]
            # Derivative contains zero: Newton step is unbounded, so bounds are unbounded
            lower_bound[i] = -Inf
            upper_bound[i] = Inf
        end
    end

    return lower_bound, upper_bound
end

function contract_newton(
    c̃::AbstractVector{<:Union{Complex{T},T}},
    λ::AbstractVector{<:Union{Complex{T},T}},
    h̃::T,
    t_inf::T,
    t_sup::T
) where {T<:AbstractFloat}

end


function refine(orig_idx, c̃, λ, h̃, t_inf, t_sup; ϵ_zero=1e-8, δ_zero=1e-8)
    @assert t_inf <= t_sup "t_inf must be the lower bound, the interval is empty/not valid"
    f, df = t -> real(ContPLRNNSolution(orig_idx, t, c̃, λ, h̃)), t -> real(ContPLRNNSolutionDerivative(orig_idx, t, c̃, λ))
    f_inf, f_sup = f(t_inf), f(t_sup)
    while t_sup - t_inf > ϵ_zero
        if f_inf * f_sup < 0
            return find_zero((f, df), (t_inf, t_sup))
        elseif abs(f_sup) < δ_zero
            return t_sup
        elseif abs(f_inf) < δ_zero
            return t_inf
        else
            contract_lower_bound, contract_upper_bound = contract_newton(@view(c̃[[orig_idx], :]), λ, @view(h̃[[orig_idx]]), t_inf, t_sup)

            #intersect the interval with the original interval
            NX_lower_bound, NX_upper_bound = intersect_interval(t_inf, t_sup, contract_lower_bound, contract_upper_bound)
            t_inf, t_sup = NX_lower_bound[1], NX_upper_bound[1]
        end
    end
    return t_inf
end

"""
process needed for the BranchAndPrune Search. Evaluates a Newton Step in the Rootfinding scheme
Modelled after contract (IntervalRootFinding.jl/contractors.jl)
"""

function contract_interval_newton(c̃::AbstractMatrix, λ::AbstractVector, h̃::AbstractVector, root_multiple::RootMultiple; kwargs...)
    if isreal(λ)
        return contract_interval_newton(Float32.(c̃), Float32.(λ), Float32.(h̃), root_multiple; kwargs...)
    else
        return contract_interval_newton(ComplexF32.(c̃), ComplexF32.(λ), Float32.(h̃), root_multiple; kwargs...)
    end
end

function contract_interval_newton(
    c̃::AbstractMatrix{<:Union{Complex{T},T}},
    λ::AbstractVector{<:Union{Complex{T},T}},
    h̃::AbstractVector{<:T},
    root_multiple::RootMultiple;
    max_region_width::T=5.0f0,
    ϵ_zero::T=1e-8,
    δ_zero::T=1e-8,
    kwargs...,
) where {T<:AbstractFloat}
    t_inf, t_sup = root_multiple.region
    id_active = root_multiple.id_active

    if t_sup - t_inf > max_region_width
        return :shrink, RootMultiple((t_inf, t_sup), id_active), (T(Inf), 0)
    end

    #only consider the dimensions, which are not pruned yet
    c̃_active, h̃_active = c̃[id_active, :], h̃[id_active]
    n_active = length(id_active)

    root_value = fill(T(Inf), n_active)  # Use T(Inf) for type consistency

    #check with bisection if image interval contains a zero
    lower_bound, upper_bound = ContPLRNNSolution_interval(c̃_active, λ, h̃_active, t_inf, t_sup)

    contains_zero_true = contains_zero(lower_bound, upper_bound)
    contains_zero_indices = id_active[contains_zero_true]

    #if all intervals contain no zero, we are done
    if isempty(contains_zero_indices)
        return :prune, RootMultiple((t_inf, t_sup), Int[]), (T(Inf), 0)
    end

    #contract the interval with Newton method
    contract_lower_bound, contract_upper_bound = contract_newton(@view(c̃[contains_zero_indices, :]), λ, @view(h̃[contains_zero_indices]), t_inf, t_sup)

    #intersect the interval with the original interval
    NX_lower_bound, NX_upper_bound = intersect_interval(t_inf, t_sup, contract_lower_bound, contract_upper_bound)

    isempty_true = isempty_interval(NX_lower_bound, NX_upper_bound)

    #if all intervals are empty, we are done
    if all(isempty_true)
        return :prune, RootMultiple((t_inf, t_sup), Int[]), (T(Inf), 0)
    end

    #determine the indices that are empty and the ones still unknown
    is_unknown_indices = contains_zero_indices[.!isempty_true]
    # Create views for non-empty NX bounds once and reuse them
    NX_lower_nonempty = @view NX_lower_bound[.!isempty_true]
    NX_upper_nonempty = @view NX_upper_bound[.!isempty_true]

    #check for the non empty intervals if they are a strict subset of the original interval
    # Reuse the views we already created
    isstrictsubset_true = isstrictsubset_interval(NX_lower_nonempty, NX_upper_nonempty, t_inf, t_sup)
    isunique_indices = is_unknown_indices[isstrictsubset_true]

    # Build lookup maps once (only if we have unique indices, to avoid overhead when not needed)
    id_active_map = nothing
    is_unknown_map = nothing
    if !isempty(isunique_indices)
        # Build lookup map for id_active to avoid repeated findfirst calls (O(1) vs O(n))
        id_active_map = Dict{Int,Int}()
        @inbounds for (pos, idx) in enumerate(id_active)
            id_active_map[idx] = pos
        end
        # Build lookup map for is_unknown_indices
        is_unknown_map = Dict{Int,Int}()
        @inbounds for (pos, idx) in enumerate(is_unknown_indices)
            is_unknown_map[idx] = pos
        end

        # Map unique indices back to positions efficiently
        @inbounds for orig_idx in isunique_indices
            pos_in_active = id_active_map[orig_idx]
            pos_in_unknown_idx = is_unknown_map[orig_idx]
            root_value[pos_in_active] = refine(orig_idx, c̃, λ, h̃, NX_lower_nonempty[pos_in_unknown_idx], NX_upper_nonempty[pos_in_unknown_idx];
                ϵ_zero=ϵ_zero, δ_zero=δ_zero)
        end
    end

    # More efficient: avoid vcat by checking t_sup separately
    # Only check indices where root_value was actually set (where we found unique roots)
    t_root_min = T(Inf)
    t_root_min_index = 0
    if !isnothing(id_active_map) && !isempty(isunique_indices)
        # Only check positions where we set root_value
        @inbounds for orig_idx in isunique_indices
            pos_in_active = id_active_map[orig_idx]
            if root_value[pos_in_active] < t_root_min
                t_root_min = root_value[pos_in_active]
                t_root_min_index = pos_in_active
            end
        end
    end

    # Adjust index: 0 means t_sup, otherwise it's the position in root_value
    t_root_min_index = t_root_min_index == 0 ? 0 : id_active[t_root_min_index]

    #all intervals are determined
    if all(isstrictsubset_true)
        return :store, RootMultiple((t_inf, t_sup), Int[]), (t_root_min, t_root_min_index)
    end

    # Optimized: compute NX bounds for non-strict-subset intervals without intermediate arrays
    # Reuse the views we already created (NX_lower_nonempty, NX_upper_nonempty)
    not_strict_subset_mask = .!isstrictsubset_true
    if any(not_strict_subset_mask)
        # Compute min/max in a single pass without creating intermediate arrays
        NX_inf = T(Inf)
        NX_sup = T(-Inf)
        @inbounds for idx in eachindex(not_strict_subset_mask)
            if not_strict_subset_mask[idx]
                NX_inf = min(NX_inf, NX_lower_nonempty[idx])
                NX_sup = max(NX_sup, NX_upper_nonempty[idx])
            end
        end
        # Use min instead of minimum([t_root_min, NX_sup]) to avoid array allocation
        return :branch, RootMultiple((NX_inf, min(t_root_min, NX_sup, t_sup)), is_unknown_indices[not_strict_subset_mask]), (t_root_min, t_root_min_index)
    else
        return :branch, RootMultiple((t_inf, min(t_root_min, t_sup)), is_unknown_indices[.!isstrictsubset_true]), (t_root_min, t_root_min_index)
    end
end

"""
Function making the process /contract function for the BranchAndPrune Search to save memory allocations?
"""

function make_contract_function(c̃_pwl, λ, h̃_pwl; kwargs...)
    return (X) -> contract_interval_newton(c̃_pwl, λ, h̃_pwl, X; kwargs...)
end

root_multiple_search(c̃_pwl, λ, h̃_pwl, root_multiple::RootMultiple; kwargs...) =
    RootMultipleSearch(
        DepthFirst,
        make_contract_function(c̃_pwl, λ, h̃_pwl; kwargs...),
        bisect_region_multiple,
        root_multiple;
        kwargs...
    )


""" 
Custom iterate for RootMultipleSearch incorporating early stopping
and tracking the smallest root seen so far. Defined on our own wrapper
type instead of the external BranchAndPruneSearch to avoid method piracy
(which breaks precompilation).
"""

function Base.iterate(
    rms::RootMultipleSearch{S},
    state=SearchState_Root(S, rms.search.initial_region);
    max_region_width=5.0f0) where S

    bp = rms.search

    search = state.search_order

    node = pop!(search)
    isnothing(node) && return nothing

    if node.region.region[1] >= state.t_root_min
        return nothing
    end

    action, region, (t_root_min, t_root_min_index) = bp.process(node.region)

    if action == :store
        node.region = region
        node.status = :final
    elseif action == :branch
        left_data, right_data = bp.bisect(region)
        node.region = nothing
        node.status = :branching
        node.left_child = BPNode(:working, left_data, node, :left)
        node.right_child = BPNode(:working, right_data, node, :right)
        push!(search, node.right_child)
        push!(search, node.left_child)
    elseif action == :prune
        prune!(node)
    elseif action == :shrink
        left_data, right_data = bisect_region_left_multiple(region, max_region_width)
        node.region = nothing
        node.status = :branching
        node.left_child = BPNode(:working, left_data, node, :left)
        node.right_child = BPNode(:working, right_data, node, :right)
        push!(search, node.right_child)
        push!(search, node.left_child)


        new_state = SearchState_Root(
            state.search_order,
            state.tree,
            state.iteration + 1,
            state.t_root_min,
            state.t_root_min_index
        )
        return new_state, new_state
    else
        error("process function for the search return " *
              "unknown action :$action for region of type $(typeof(region)). " *
              "Valid actions are :store, :branch, :prune and :shrink.")
    end

    # keep track of the best (minimal) root seen so far
    best_root = state.t_root_min
    best_idx = state.t_root_min_index
    if isfinite(t_root_min) && t_root_min < best_root
        best_root = t_root_min
        best_idx = t_root_min_index
    end

    new_state = SearchState_Root(
        state.search_order,
        state.tree,
        state.iteration + 1,
        best_root,
        best_idx
    )
    return new_state, new_state
end





function roots_multiple(c̃_pwl, λ, h̃_pwl, region; kwargs...)
    P = length(h̃_pwl)
    root_multiple = RootMultiple(region, Vector{Int}(1:P))
    search = root_multiple_search(c̃_pwl, λ, h̃_pwl, root_multiple; kwargs...)
    max_iter = kwargs[:max_iteration]
    endstate = nothing
    for (iter, state) in enumerate(search)
        endstate = state
        iter >= max_iter && break
    end
    return endstate.t_root_min, endstate.t_root_min_index
end



Zygote.@adjoint function roots_multiple(c̃_pwl, λ, h̃_pwl, region; kwargs...)
    # Forward pass: call the function with kwargs
    t_min, i_min = roots_multiple(c̃_pwl, λ, h̃_pwl, region; kwargs...)

    function pullback(Δ)
        # Early return for no root case (only check isinf, not i_min == 0)
        if isinf(t_min)
            @warn "No root found in roots_multiple (t_min is Inf), returning zero gradients."
            return (
                zeros(size(c̃_pwl)),
                zeros(length(λ)),
                zeros(length(h̃_pwl)),
                nothing  # region is typically not differentiable (Tuple)
            )
        end

        # Extract gradient for time (roots_multiple returns (t_min, i_min))
        Δt_min = Δ isa Tuple ? Δ[1] : Δ  # ∂L/∂t_min

        # Compute derivative once and cache
        dz_dt = ContPLRNNSolutionDerivative(i_min, t_min, c̃_pwl, λ)
        if abs(dz_dt) < kwargs[:dz_tangent_threshold]
            @warn "Derivative near zero at root in roots_multiple, unstable gradient. Returning zero gradients."
            return (
                zeros(size(c̃_pwl)),
                zeros(length(λ)),
                zeros(length(h̃_pwl)),
                nothing
            )
        end

        # Optimized: compute exp(λ * t_min) once and reuse
        exp_λt = @. exp(λ * t_min)  # Compute once, reuse for both dc̃_pwl and dλ

        # Compute factor once
        factor = real(-Δt_min / dz_dt)

        # Pre-allocate gradients with matching types (complex if inputs are complex)
        dc̃_pwl = zeros(eltype(c̃_pwl), size(c̃_pwl))
        @inbounds dc̃_pwl[i_min, :] = @. exp_λt * factor  # ∂t_min/∂c̃_pwl (compute directly)

        # Compute dλ efficiently: reuse exp_λt and c̃_pwl row, apply factor directly
        @inbounds c̃_row = @view c̃_pwl[i_min, :]
        dλ = @. c̃_row * t_min * exp_λt * factor  # ∂t_min/∂λ (compute directly)

        # Pre-allocate dh̃_pwl with matching type
        dh̃_pwl = zeros(eltype(h̃_pwl), length(h̃_pwl))
        @inbounds dh̃_pwl[i_min] = factor  # ∂t_min/∂h̃_pwl (1.0 * factor = factor)

        # Return gradients for positional arguments only: (c̃_pwl, λ, h̃_pwl, region)
        # kwargs are automatically handled by Zygote and don't need to be returned
        # region is typically a Tuple and not differentiable, so return nothing
        return (dc̃_pwl, dλ, dh̃_pwl, nothing)
    end

    return (t_min, i_min), pullback
end
