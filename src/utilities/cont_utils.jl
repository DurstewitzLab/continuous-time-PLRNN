using LinearAlgebra
using Random




#Define outer product to compute quantites for vectors and matrices
outprod(a::AbstractVector, b::AbstractVector) = repeat(a, 1, length(b)).*transpose(repeat(b, 1, length(a)))
function outprod(a::AbstractVector, B::AbstractMatrix)
    @assert length(a)==size(B,1)
    repeat(a, 1, size(B, 2)).*B
end
#Comparable to outprod
outsum(a::AbstractVector, b::AbstractVector) = repeat(a, 1, length(b)).+transpose(repeat(b, 1, length(a)))

#expression of f(t) in sum representation
ContPLRNNSolution(tvec::AbstractVector, c̃::AbstractMatrix, λ::AbstractVector, h̃::AbstractVector) = c̃*exp.(outprod(λ,tvec))+repeat(h̃,1,length(tvec))
#for different dimensions i
ContPLRNNSolution(i::Integer, t::Union{Real, Complex}, c̃::AbstractMatrix, λ::AbstractVector, h̃::AbstractVector) = sum(c̃[i,:] .* exp.(λ*t))+h̃[i]

#derivatives => needed for RootFinderInterval as roots needs the derivative
ContPLRNNSolutionDerivative(i::Integer, t::Union{Real,Complex}, c̃::AbstractMatrix, λ::AbstractVector) = sum(λ.* c̃[i,:] .* exp.(λ*t))
ContPLRNNSolutionDerivative(i::Integer, t::Complex, c̃::AbstractMatrix, λ::AbstractVector) = sum(λ.* c̃[i,:] .* exp.(λ*t))

#sum representation given W, z0 and h
function SumRep(W::AbstractMatrix, z0::AbstractVector, h::AbstractVector)
    M = size(W, 1)
    λ, P = eigen(W)
    Pinv = inv(P)
    h̃ = - inv(W)*h
    c = Pinv*(z0 .- h̃)
    c̃ = P.*transpose(repeat(c, 1, M)) #' is also complex conjugate, do not use it!!!
    return c̃, λ, h̃
end


using Zygote
using IntervalArithmetic.Symbols #needed to be able to use ..
using IntervalRootFinding
using Roots: find_zero  

# Add these at the top of your file
Zygote.@nograd roots
Zygote.@nograd minimum
Zygote.@nograd inf


#function to find the minimum of the switching times
function RootFinderInterval(a::Real, b::Real, f::Function, df::Function, M::Int)    
    tswitch_vec = zeros(M)
    for i in 1:M
        g_i = (t) -> real.(f(i, t))
        dg_i = (t) -> real.(df(i, t))
        root_array = roots(g_i, a..b, derivative = dg_i) #; contractor = Newton#Can give exact derivative, because one of the neurode packages
        #also has Newton
        tswitch_vec[i] = (isempty(root_array) ? Inf : minimum(inf.([r.region for r in root_array])))
    end
    min_value, min_index = findmin(tswitch_vec) 
    return (Float32(min_value), min_index) #did not find a solution doing it in one line with the findmin
end



using ThreadsX
function RootFinderInterval(a::Float32, b::Real, f::Function, df::Function, M::Int, P::Int)
    #tswitch_vec = zeros(P)
    tswitch_vec = fill(Inf32, P)

    # Pre-allocate functions
    g_functions = Vector{Function}(undef, P)
    dg_functions = Vector{Function}(undef, P)
    
    # Pre-define functions to avoid recreation in each iteration
    for (idx, i) in enumerate(M-P+1:M)
        g_functions[idx] = t -> real.(f(i, t))
        dg_functions[idx] = t -> real.(df(i, t))
    end

    # Use ThreadsX for parallel processing
    ThreadsX.foreach(M-P+1:M) do i
        idx = i - (M-P)
        g_i = g_functions[idx] #(t) -> real.(f(i, t))
        dg_i = dg_functions[idx] #(t) -> real.(df(i, t))
        
        t1 = time_ns()
        #root_array = roots(g_i, a..b, derivative = dg_i, max_iteration = 1000) #; contractor = Newton#Can give exact derivative, because one of the neurode packages
        t2 = time_ns()
        #if (t2-t1)/1e9 > 0.04
        #println("i: $i, time for roots: $((t2-t1)/1e9), roots: $(length(root_array)), unique: $(any(r -> r.status == :unique, root_array))")
        #end
        #println("root_array: $root_array")
        # Filter for unique roots only
        #root_array = filter(r -> r.status == :unique, root_array) #probably need only first entry
        #also has Newton
        t3 = time_ns()
        #root_array = filter(r -> inf(r.region) > 0.001f0, root_array)
        tswitch_vec[i-(M-P)] = find_first_root(g_i, dg_i, a, b) #(isempty(root_array) ? Inf : minimum([RootFinder(g_i, dg_i, inf(r.region), sup(r.region)) for r in root_array]))
        t4 = time_ns()
        #println("time for rootfinder: $((t4-t3)/1e9), tswitch_vec: $(tswitch_vec[i-(M-P)])")
        #tswitch_vec[i-(M-P)] = (isempty(root_array) ? Inf : minimum(inf.([r.region for r in root_array])))
    end
    min_value, min_index = findmin(tswitch_vec) 
    return (Float32(min_value), min_index)
end

function find_first_root(g::Function, dg::Function, a::Real, b::Real)
    root_array = roots(g, a..b, derivative=dg, max_iteration=1000, search_order = DepthFirst)
    unique_roots = filter(r -> r.status == :unique, root_array)
    
    if isempty(unique_roots)
        return Inf 
    else
        # Extract roots efficiently
        return minimum([RootFinder(g, dg, inf(r.region), sup(r.region)) for r in unique_roots])
    end

end


# #function to find the minimum of the switching times for the ALRNN, additional Parameter P
# function RootFinderInterval(a::Real, b::Real, f::Function, df::Function, M::Int, P::Int; previous_i::Int = 0, Δt::Float32 = 0.001f0)
#     tswitch_vec = zeros(P)
#     dim_vec = zeros(Int, P)
#     for i in M-P+1:M
#         g_i = (t) -> real.(f(i, t))
#         dg_i = (t) -> real.(df(i, t))
#         roots_not_unique = true
#         max_iteration = 2
#         root_array = Root{Interval{Float32}}[]
#         while roots_not_unique && max_iteration > 0 #if the roots are not unique, it can be due to max_iteration not feasible for upper limit b
#             #(possible zero derivative at b?) and it can help to change b slightly.
#             println("a: $a, b: $b")
#             println("g_i(a): $(g_i(a)), g_i(b): $(g_i(b))")
#             root_array = roots(g_i, a..b, derivative = dg_i, max_iteration = 1000) #; contractor = Newton#Can give exact derivative, because one of the neurode packages
#             println("root_array: $root_array")
#             roots_not_unique = any(r.status != :unique for r in root_array)
#             b = b + 0.1f0
#             max_iteration -= 1
#             if max_iteration == 0
#                 @warn "RootFinderInterval: Max iteration reached for i: $i"
#             end
#         end
#         root_array = roots(g_i, a..b, derivative = dg_i, max_iteration = 1000)
#         #root_point_array = [RootFinder(g_i, dg_i, inf(r.region), sup(r.region)) for r in root_array]
#         #root_point_array = (i == previous_i + M - P ? filter(x -> x > Δt, root_point_array) : root_point_array)
#         #also has Newton
#         tswitch_vec[i-(M-P)], dim_vec[i-(M-P)] = (isempty(root_array) ? (Inf, 0) : findmin([RootFinder(g_i, dg_i, inf(r.region), sup(r.region)) for r in root_array]))
#         #tswitch_vec[i-(M-P)] = (isempty(root_array) ? Inf : minimum(inf.([r.region for r in root_array])))
#     end
#     min_value, min_index = findmin(tswitch_vec) 
#     return (Float32(min_value), min_index)
# end


# function to find the minimum of the switching times for the ALRNN, additional Parameter P
# function RootFinderInterval(a::Real, b::Real, f::Function, df::Function, M::Int, P::Int; previous_i::Int = 0)
#     tswitch_vec = zeros(Float32, P)
#     Δt = 0.0f0 #0.001f0
#     for i in M-P+1:M
#         g_i = (t) -> real.(f(i, t))
#         dg_i = (t) -> real.(df(i, t))
#         root_array = roots(g_i, a..b, derivative = dg_i) #; contractor = Newton#Can give exact derivative, because one of the neurode packages
#         #also has Newton
#         if i == previous_i + M - P
#             if !isempty(root_array)
#                 root_1 = root_array[1]
#                 root_array = (inf(root_array[1].region) < Δt ? root_array[2:end] : root_array)
#                 #println("root_array previous: $root_array")
#             end
#         end
#         #println("root_array $(i): $root_array")
#         tswitch_vec[i-(M-P)] = (isempty(root_array) ? Inf : minimum([RootFinder(g_i, dg_i, inf(r.region), sup(r.region)) for r in root_array]))
#         #tswitch_vec[i-(M-P)] = (isempty(root_array) ? Inf : minimum(inf.([find_zero(g_i, (inf(r.region), sup(r.region)), derivative = dg_i) for r in root_array])))
#     end
#     min_value, min_index = findmin(tswitch_vec) 
#     return (Float32(min_value), min_index)
# end

# #function to find the minimum of the switching times for the ALRNN, additional Parameter P
# function RootFinderInterval(a::Real, b::Real, f::Function, df::Function, M::Int, P::Int; previous_i::Int = 0)
#     tswitch_vec = zeros(Float32, P)
#     dim_vec = zeros(Int, P)
#     infsup_vec = zeros(Float32, P, 2)
#     Δt = 0.001f0
#     for i in M-P+1:M
#         g_i = (t) -> real.(f(i, t))
#         dg_i = (t) -> real.(df(i, t))
#         root_array = roots(g_i, a..b, derivative = dg_i)
#         for r in root_array
#             r.status == :unique ? nothing : println(r.status, g_i(inf(r.region)), g_i(sup(r.region)))
#         end #; contractor = Newton#Can give exact derivative, because one of the neurode packages
#         #also has Newton
#         if i == previous_i + M - P
#             if !isempty(root_array)
#                 root_1 = root_array[1]
#                 root_array = (inf(root_array[1].region) < Δt ? root_array[2:end] : root_array)
#                 #println("root_array previous: $root_array")
#             end
#         end
#         #println("root_array $(i): $root_array")
#         tswitch_vec[i-(M-P)], dim_vec[i-(M-P)] = (isempty(root_array) ? (Inf, 0) : findmin([RootFinder(g_i, dg_i, inf(r.region), sup(r.region)) for r in root_array]))
#         infsup_vec[i-(M-P), :] .= (iszero(dim_vec[i-(M-P)]) ? (Inf, Inf) : (inf(root_array[dim_vec[i-(M-P)]].region), sup(root_array[dim_vec[i-(M-P)]].region)))
#         #tswitch_vec[i-(M-P)] = (isempty(root_array) ? Inf : minimum(inf.([find_zero(g_i, (inf(r.region), sup(r.region)), derivative = dg_i) for r in root_array])))
#     end
#     min_value, min_index = findmin(tswitch_vec)
#     println("min_value: $min_value, min_index: $min_index, min_func_val: $(f(min_index+M-P, min_value))") 
#     return (Float32(min_value), min_index, infsup_vec[min_index, :]...)
# end


function RootFinder(g_i::Function, dg_i::Function, a::Real, b::Real)
    tolerance = 0.0f0#1e-6
    if g_i(a)*g_i(b) < 0
        return find_zero(g_i, (a,b), derivative = dg_i)
    elseif abs(g_i(a)) <= tolerance  # Changed from g_i(a) ≈ 0
        if abs(g_i(b)) <= tolerance  # Changed from g_i(b) ≈ 0
            #@warn "Constantly zero"
            #println("dg_i(a): $(dg_i(a)), dg_i(b): $(dg_i(b))")
            return Inf
        else
            #@warn "RootFinder: Root found at a: $(a)"
            #println(g_i(a-(b-a)), " ", g_i(a), " ", g_i(b), " ", g_i(a+(b-a)))
            return a
        end
    elseif abs(g_i(b)) <= tolerance
        #@warn "RootFinder: Root found at b: $(b)"
        #println(g_i(a), " ", g_i(b), " ", g_i(b+(b-a)))
        return b
    else #cannot resolve the root in the interval, treat it as non-existing
        #@warn "Cannot resolve the root in the interval $(a) to $(b), treat it as non-existing"
        #println("g_i(a): $(g_i(a)), g_i((a+b)/2): $(g_i((a+b)/2)), g_i(b): $(g_i(b))")
        return (a+b)/2
    end
end

Zygote.@nograd RootFinderInterval
Zygote.@nograd RootFinder


function compute_plrnn_trajectory(
    t_points::AbstractVector,
    A::AbstractVecOrMat,
    W::AbstractMatrix, 
    h::AbstractVector, 
    z0::AbstractVector;
    max_iterations::Int = 100,
    Δt::Float32 = 0.01f0,#0.0f0, #0.0001f0,
    save_switching_times::Bool = false
)   

    t_points = t_points.-t_points[1] #shift t_points to start at 0
    tmax = t_points[end] #are already assumed to be sorted #maximum(t_points)
    M = size(z0, 1)

    # Initialize output storage
    z_out = reshape(Float32[], M, 0)
    idx_begin = 1  # Track position in t_points
    idx_end = 1
    ttotal = 0.0f0
    iterations = 0
    #Storage for switching times
    if save_switching_times
        t_switch_vec = Float32[]
        dim_switch_vec = Int[]
    end

    # Initial system
    z_current = z0
    Wsub = A + W * Diagonal(z_current .> 0)
    c̃, λ, h̃ = SumRep(Wsub, z_current, h)
    Δt2 = 0.0f0
    
    while idx_begin <= length(t_points) && iterations < max_iterations
        #&& ttotal < tmax #had this requiremnt in the while loop, but I do not now why...Probably to avoid the case of Δt bringing ttotal to be bigger than tmax
        # Find next switching time
        t_switch, dim_switch = RootFinderInterval(Δt2, tmax - ttotal, 
            (i,t) -> ContPLRNNSolution(i,t,c̃,λ,h̃), 
            (i,t) -> ContPLRNNSolutionDerivative(i,t,c̃,λ), 
            M)
        #save switching times and dimensions
        #if save_switching_times #not possible if wanted differentiable
        #    # in the first iteration we do not shift the switching time
        #    iterations > 0 ? push!(t_switch_vec, t_switch+Δt) : push!(t_switch_vec, t_switch)
        #    push!(dim_switch_vec, dim_switch)
        #end
        # Find range of t_points in current linear region
        region_end = ttotal + t_switch
        # Search only in remaining portion and adjust by idx_begin - 1 to get global index
        idx_end = idx_begin + searchsortedlast(view(t_points, idx_begin:length(t_points)), region_end) - 1

        # Compute local times for all points in current region
        t_local = t_points[idx_begin:idx_end] .- ttotal
        
        # Compute solution for all points at once
        #Cannot do something like z_out[:, idx_begin:idx_end] = z_local since mutating 
        #a vector is not possible in Zygote. Therefore: Concatenating
        z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
        z_out = hcat(z_out, z_local)

        idx_begin = idx_end + 1

        # Exit if we've processed all points or reached final region
        if idx_begin > length(t_points) || t_switch == Inf
            break
        end

        # Prepare for next iteration
        ttotal += t_switch + Δt
        z_current = real.(ContPLRNNSolution([t_switch + Δt], c̃, λ, h̃)[:])
        Wsub = A + W * Diagonal(z_current .> 0)
        c̃, λ, h̃ = SumRep(Wsub, z_current, h)
        Δt2 = 0.001f0

        ### This new break hast to be tested and checked
        if tmax-ttotal < Δt #if the next switching time is less than Δt, we cannot compute the next 
            #switching time. and have to assume that the system is in the last region
            t_local = t_points[idx_begin:end] .- ttotal
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
            break
        end

        # Compute the next switching time
        iterations += 1
        if iterations == max_iterations
            println("Maximum iterations reached")
        end
    end
    return z_out #save_switching_times ? (z_out, t_switch_vec, dim_switch_vec) : z_out
end

function compute_alrnn_trajectory(
    t_points::AbstractVector,
    A::AbstractVecOrMat,
    W::AbstractMatrix, 
    h::AbstractVector, 
    z0::AbstractVector,
    P::Int;
    max_iterations::Int = 10000,
    Δt::Float32 = 0.0f0, #0.001f0,
    save_switching_times::Bool = false, 
    stop_flag::Bool = false
)   

    t_points = t_points.-t_points[1] #shift t_points to start at 0
    tmax = t_points[end] #are already assumed to be sorted #maximum(t_points)
    M = size(z0, 1)


    #if A is a vector, make it a diagonal matrix
    if A isa AbstractVector
        A = Diagonal(A)
    end

    # Initialize output storage
    z_out = reshape(Float32[], M, 0)
    idx_begin = 1  # Track position in t_points
    idx_end = 1
    ttotal = 0.0f0
    iterations = 0
    id_switch = 0
    Δt2 = 0.0f0
    #Storage for switching times
    if save_switching_times
        t_switch_vec = Float32[]
        dim_switch_vec = Int[]
    end

    # Initial system
    z_current = z0
    Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_current[end-(P-1):end] .> 0))
    h_trafo = Wsub*inv(I-exp(Wsub))*h
    #in the future, maybe ake this nice and dependent on general type T
    c̃, λ, h̃ = SumRep(Wsub, z_current, h_trafo)
    
    while idx_begin <= length(t_points) && iterations < max_iterations
        #println("iterations: $iterations, ttotal: $ttotal, tmax: $tmax")
        #&& ttotal < tmax #had this requiremnt in the while loop, but I do not now why...
        # Find next switching time

        t₁ = time_ns()

        t_switch, dim_switch = RootFinderInterval(Δt2, tmax - ttotal, 
        (i,t) -> ContPLRNNSolution(i,t,c̃,λ,h̃), 
        (i,t) -> ContPLRNNSolutionDerivative(i,t,c̃,λ), 
        M, P) #, previous_i=id_switch

        t_switch, dim_switch = Zygote.dropgrad(t_switch), Zygote.dropgrad(dim_switch)
        t₂ = time_ns()
        #println("t_switch: $t_switch, dim_switch: $dim_switch, time for Rootfinding: $((t₂-t₁)/1e9)")
        #save switching times and dimensions
        #if save_switching_times #not possible if wanted differentiable
        #    # in the first iteration we do not shift the switching time
        #    iterations > 0 ? push!(t_switch_vec, t_switch+Δt) : push!(t_switch_vec, t_switch)
        #    push!(dim_switch_vec, dim_switch)
        #end
        # Find range of t_points in current linear region
        region_end = ttotal + t_switch
        # Search only in remaining portion and adjust by idx_begin - 1 to get global index
        idx_end = idx_begin + searchsortedlast(view(t_points, idx_begin:length(t_points)), region_end) - 1

        # Compute local times for all points in current region
        t_local = t_points[idx_begin:idx_end] .- ttotal
        
        # Compute solution for all points at once
        #Cannot do something like z_out[:, idx_begin:idx_end] = z_local since mutating 
        #a vector is not possible in Zygote. Therefore: Concatenating
        z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
        z_out = hcat(z_out, z_local)

        idx_begin = idx_end + 1

        # Exit if we've processed all points or reached final region
        if idx_begin > length(t_points) || t_switch == Inf
            break
        end

        # Prepare for next iteration
        ttotal += t_switch + Δt
        #println("ttotal: $ttotal")
        z_current = real.(ContPLRNNSolution([t_switch + Δt], c̃, λ, h̃)[:])
        z_diag = real.(ContPLRNNSolution([t_switch + 0.0001f0], c̃, λ, h̃)[:]) #if this is useful, can be combined with z_current
        Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_diag[end-(P-1):end] .> 0))
        h_trafo = Wsub*inv(I-exp(Wsub))*h
        c̃, λ, h̃ = SumRep(Wsub, z_current, h_trafo)
        id_switch = dim_switch
        Δt2 = 0.0001f0
        #println("id_switch: $id_switch")
    

        ### This new break hast to be tested and checked
        if tmax-ttotal < Δt #if the next switching time is less than Δt, we cannot compute the next 
            #switching time. and have to assume that the system is in the last region
            t_local = t_points[idx_begin:end] .- ttotal
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
            break
        end

        # Compute the next switching time
        iterations += 1
        if iterations == max_iterations
            @warn "Maximum iterations $(max_iterations) of the trajectory computation reached.
            The system is changing linear subregions to often."
            z_local = ones(Float32, M, length(t_points[idx_begin:end]))*NaN#real.(ContPLRNNSolution([t_points[end]], c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
            stop_flag = true
            #println("Maximum iterations reached")
        end
    end
    return z_out#save_switching_times ? (z_out, t_switch_vec, dim_switch_vec) : z_out
end

"""
    compute_alrnn_trajectory_cached

Optimized ALRNN trajectory computation with matrix caching.
This version caches matrix operations to avoid redundant computations
when the same switching patterns are encountered multiple times.

# Arguments
- `t_points`: Time points for trajectory computation
- `A`: Diagonal matrix or vector for linear dynamics
- `W`: Weight matrix for nonlinear dynamics  
- `h`: Bias vector
- `z0`: Initial condition
- `P`: Number of piecewise linear units

# Keyword Arguments
- `max_iterations`: Maximum number of iterations (default: 10000)
- `Δt`: Time step for switching detection (default: 0.0f0)
- `Δt_diag`: Diagonal time step (default: 0.00001f0)
- `save_switching_times`: Whether to save switching times (default: false)
- `stop_flag`: Stop flag (default: false)
- `max_t_interval`: Maximum time interval (default: 1000.0f0)
- `use_cache`: Whether to use matrix caching (default: true)
- `batch_size`: Batch size for solution computation (default: 1000)
- `verbose`: Whether to print verbose output (default: false)

# Returns
- `z_out`: Computed trajectory matrix
"""
function compute_alrnn_trajectory_cached(
    t_points::AbstractVector,
    A::AbstractVecOrMat,
    W::AbstractMatrix, 
    h::AbstractVector, 
    z0::AbstractVector,
    P::Int;
    max_iterations::Int = 10000,
    Δt::Float32 = 0.0f0,
    Δt_diag::Float32 = 0.00001f0,
    save_switching_times::Bool = false, 
    stop_flag::Bool = false,
    max_t_interval::Float32 = 1000.0f0,
    use_cache::Bool = true,
    batch_size::Int = 1000,
    verbose::Bool = false
)   
    # Pre-compute constants
    t_points = t_points.-t_points[1]
    tmax = t_points[end]
    M = size(z0, 1)
    I_matrix = Matrix{Float32}(I, M, M)

    if A isa AbstractVector
        A = Diagonal(A)
    end

    # Pre-allocate output array
    z_out = zeros(Float32, M, length(t_points))
    current_idx = 1
    
    # Local matrix cache (not global like in training)
    matrix_cache = use_cache ? Dict{String, Tuple{Any, Any, Any}}() : Dict{String, Tuple{Any, Any, Any}}()
    
    idx_begin = 1
    idx_end = 1
    ttotal = 0.0f0
    iterations = 0
    id_switch = 0
    Δt2 = 0.0f0
    
    if save_switching_times
        t_switch_vec = Float32[]
        dim_switch_vec = Int[]
    end

    # Optimized matrix operations function with caching
    function compute_matrix_operations_cached(z_current, z_diag)
        indicator = vcat(ones(Float32, M-P), z_diag[end-(P-1):end] .> 0)
        indicator_key = join(indicator, "_")
        
        # Check cache first
        if use_cache && haskey(matrix_cache, indicator_key)
            return matrix_cache[indicator_key]
        end
        
        # Compute expensive matrix operations
        Wsub = A + W * Diagonal(indicator)
        
        # Use LU decomposition instead of direct inverse
        F = lu(I_matrix - Wsub)
        logWsub = log(Wsub)
        logh = -logWsub * (F \ h)  # More efficient than inv(I-Wsub)*h
        
        result = (logWsub, logh, F)
        
        # Store in cache for future use
        if use_cache
            matrix_cache[indicator_key] = result
        end
        
        return result
    end
    
    # Optimized solution computation with batching
    function optimized_solution_computation(t_local, c̃, λ, h̃)
        n_points = length(t_local)
        if n_points == 0
            return zeros(Float32, M, 0)
        end
        
        # Process in batches for better cache efficiency
        z_local = zeros(Float32, M, n_points)
        for i in 1:batch_size:n_points
            end_idx = min(i + batch_size - 1, n_points)
            batch_t = t_local[i:end_idx]
            z_local[:, i:end_idx] = real.(ContPLRNNSolution(batch_t, c̃, λ, h̃))
        end
        
        return z_local
    end

    # Initial setup
    z_current = z0
    z_diag = z_current
    logWsub, logh, F = compute_matrix_operations_cached(z_current, z_diag)
    z_indicator = z_current[end-(P-1):end] .> 0
    c̃, λ, h̃ = SumRep(logWsub, z_current, logh)
    
    while idx_begin <= length(t_points) && iterations < max_iterations
        if verbose && iterations % 100 == 0
            println("Cached iteration $iterations: ttotal=$ttotal, tmax=$tmax")
        end
        
        # Root finding (same as original)
        t_switch, dim_switch = RootFinderInterval(Δt2, min(tmax - ttotal, max_t_interval), 
            (i,t) -> ContPLRNNSolution(i,t,c̃,λ,h̃), 
            (i,t) -> ContPLRNNSolutionDerivative(i,t,c̃,λ), 
            M, P)
        t_switch, dim_switch = Zygote.dropgrad(t_switch), Zygote.dropgrad(dim_switch)

        flag_t_switch = true
        if t_switch == Inf
            t_switch = tmax - ttotal
            flag_t_switch = false
        end

        # Memory operations
        if save_switching_times
            iterations > 0 ? push!(t_switch_vec, t_switch+Δt) : push!(t_switch_vec, t_switch)
            push!(dim_switch_vec, dim_switch)
        end
        
        # Find range and compute solution
        region_end = ttotal + t_switch
        idx_end = idx_begin + searchsortedlast(view(t_points, idx_begin:length(t_points)), region_end) - 1
        t_local = t_points[idx_begin:idx_end] .- ttotal
        
        # Use optimized solution computation
        z_local = optimized_solution_computation(t_local, c̃, λ, h̃)
        
        # Use pre-allocated array instead of concatenation
        if !isempty(z_local)
            z_out[:, current_idx:current_idx+size(z_local,2)-1] = z_local
            current_idx += size(z_local, 2)
        end

        idx_begin = idx_end + 1

        if idx_begin > length(t_points) || t_switch == Inf
            break
        end

        ttotal += t_switch + Δt
        z_current = real.(ContPLRNNSolution([t_switch + Δt], c̃, λ, h̃)[:])

        # Optimized switching detection
        exp = 0
        switch_flag = false
        Δt2 = 0.0f0

        while exp<3&&switch_flag == false&&flag_t_switch == true
            z_diag = real.(ContPLRNNSolution([t_switch + Δt_diag*10^exp], c̃, λ, h̃)[:])
            if z_indicator == (z_diag[end-(P-1):end] .> 0)
                switch_flag = false
                Δt2 = Δt_diag*10^exp*2
            else
                switch_flag = true
                Δt2 = Δt_diag*10^exp*2
            end
            exp += 1
        end

        z_indicator = z_diag[end-(P-1):end] .> 0
        
        # Use cached matrix operations
        logWsub, logh, F = compute_matrix_operations_cached(z_current, z_diag)
        c̃, λ, h̃ = SumRep(logWsub, z_current, logh)
        id_switch = dim_switch

        if tmax-ttotal < Δt
            t_local = t_points[idx_begin:end] .- ttotal
            z_local = optimized_solution_computation(t_local, c̃, λ, h̃)
            if !isempty(z_local)
                z_out[:, current_idx:current_idx+size(z_local,2)-1] = z_local
            end
            break
        end

        iterations += 1
        if iterations == max_iterations
            @warn "Maximum iterations $(max_iterations) reached."
            z_out[:, current_idx:end] .= NaN
            stop_flag = true
        end
    end
    
    # Return only the used portion of the pre-allocated array
    return z_out[:, 1:current_idx-1]
end

"""
    get_cache_stats_local(cache)

Get statistics about a local matrix cache.
"""
function get_cache_stats_local(cache::Dict{String, Tuple{Any, Any, Any}})
    return Dict(
        "cache_size" => length(cache),
        "cache_keys" => collect(keys(cache))
    )
end