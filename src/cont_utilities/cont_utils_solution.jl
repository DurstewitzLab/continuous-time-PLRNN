using Base.Threads
using LinearAlgebra
using Random



function compute_alrnn_trajectory(
    t_points::AbstractVector,
    A::AbstractVecOrMat,
    W::AbstractMatrix, 
    h::AbstractVector,
    z0::AbstractVector,
    P::Int;
    max_iterations::Int = 10000,
    Δt_diag::Float32 = 0.0001f0,
    Δt_switch::Float32 = 0.0001f0,
    max_region_width::Float32 = Inf32,
    dz_tangent_threshold::Float32 = Float32(1e-12),
    ϵ_zero::Float32 = Float32(1e-8),
    δ_zero::Float32 = Float32(1e-8),
    λ_gap_threshold::Float32 = Float32(1e-10),
    dW_strength::Float32 = Float32(1e-6),
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
    Δt_switch_local = 0.0f0 #for the first iteration we do not use a reduced search interval

    # Initial system
    z_current = z0
    Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_current[end-(P-1):end] .> 0))
    c̃, λ, h̃ = SumRep_sort(Wsub, z_current, h; λ_gap_threshold=λ_gap_threshold, dW_strength=dW_strength)
    c̃_pwl, h̃_pwl = @view(c̃[M-P+1:end,:]), @view(h̃[M-P+1:end])
    
    while idx_begin <= length(t_points) && iterations <= max_iterations

        t_switch, dim_switch = roots_multiple(c̃_pwl, λ, h̃_pwl, (Δt_switch_local, tmax - ttotal);
        max_iteration=max_iterations,
        max_region_width=max_region_width,
        dz_tangent_threshold=dz_tangent_threshold,
        ϵ_zero=ϵ_zero, δ_zero=δ_zero)

        # Find range of t_points in current linear region
        region_end = ttotal + t_switch
        # Search only in remaining portion and adjust by idx_begin - 1 to get global index
        idx_end = idx_begin + searchsortedlast(view(t_points, idx_begin:length(t_points)), region_end) - 1

        if idx_begin <= idx_end #points in current region whithout this the derivative had the wrong dimension
            # Compute local times for all points in current region
            t_local = t_points[idx_begin:idx_end] .- ttotal
            # Compute solution for all points at once   
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
        end

        idx_begin = idx_end + 1

        # Exit if we've processed all points or reached final region or we would reach the maximum number of iterations
        if idx_begin > length(t_points) || t_switch == Inf || iterations == max_iterations
            break
        end

        # Prepare for next iteration
        ttotal += t_switch
        z_current = real.(ContPLRNNSolution([t_switch], c̃, λ, h̃)[:])
        z_diag = real.(ContPLRNNSolution([t_switch + Δt_diag], c̃, λ, h̃)[:]) #if this is useful, can be combined with z_current
        Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_diag[end-(P-1):end] .> 0))
        c̃, λ, h̃ = SumRep_sort(Wsub, z_current, h)
        c̃_pwl, h̃_pwl = c̃[M-P+1:end,:], h̃[M-P+1:end]
        id_switch = dim_switch
        
        Δt_switch_local = Δt_switch #now use open interval for the next switching time

        ### Edge case: 
        if tmax-ttotal < Δt_switch_local #if the next switching time is less than Δt, we cannot compute the next 
            #switching time. and have to assume that the system is in the last region
            t_local = t_points[idx_begin:end] .- ttotal
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
            break
        end

        # Compute the next switching time
        iterations += 1

    end
    return z_out
end




function compute_alrnn_trajectory_fully_cached(
    cache::BatchCache{Any,Any},
    t_points::AbstractVector,
    A::AbstractVecOrMat,
    W::AbstractMatrix, 
    h::AbstractVector,
    z0::AbstractVector,
    P::Int;
    max_iterations::Int = 10000,
    Δt_diag::Float32 = 0.0001f0,
    Δt_switch::Float32 = 0.0001f0,
    max_region_width::Float32 = Inf32,
    dz_tangent_threshold::Float32 = Float32(1e-12),
    ϵ_zero::Float32 = Float32(1e-8),
    δ_zero::Float32 = Float32(1e-8),
    save_switching_times::Bool = false,
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
    Δt_switch_local = 0.0f0 #for the first iteration we do not use a reduced search interval

    #Storage for switching times
    if save_switching_times
        t_switch_vec = Float32[]
        dim_switch_vec = Int[]
    end
    
    # Initial system
    z_current = z0
    Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_current[end-(P-1):end] .> 0))
    c̃, λ, h̃ = SumRep_sort_fully_cached(cache,A, W, h, z_current, z0, M, P)
    h̃ = Vector{Float32}(h̃)
    c̃_pwl, h̃_pwl = @view(c̃[M-P+1:end,:]), @view(h̃[M-P+1:end])
    
    while idx_begin <= length(t_points) && iterations < max_iterations

        t_switch, dim_switch = roots_multiple(c̃_pwl, λ, h̃_pwl, (Δt_switch_local, tmax - ttotal),
        max_iteration=max_iterations,
        max_region_width=max_region_width,
        dz_tangent_threshold=dz_tangent_threshold,
        ϵ_zero=ϵ_zero, δ_zero=δ_zero)

        if save_switching_times&&t_switch != Inf 
            # in the first iteration we do not shift the switching time
            iterations > 0 ? push!(t_switch_vec, t_switch+ttotal) : push!(t_switch_vec, t_switch+ttotal)
            push!(dim_switch_vec, dim_switch)
        end

        # Find range of t_points in current linear region
        region_end = ttotal + t_switch
        # Search only in remaining portion and adjust by idx_begin - 1 to get global index
        idx_end = idx_begin + searchsortedlast(view(t_points, idx_begin:length(t_points)), region_end) - 1

        if idx_begin <= idx_end #points in current region whithout this the derivative had the wrong dimension
            # Compute local times for all points in current region
            t_local = t_points[idx_begin:idx_end] .- ttotal
            # Compute solution for all points at once   
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
        end

        idx_begin = idx_end + 1

        # Exit if we've processed all points or reached final region or we would reach the maximum number of iterations
        if idx_begin > length(t_points) || t_switch == Inf || iterations == max_iterations
            break
        end

        # Prepare for next iteration
        ttotal += t_switch
        z_current = real.(ContPLRNNSolution([t_switch], c̃, λ, h̃)[:])
        z_diag = real.(ContPLRNNSolution([t_switch + Δt_diag], c̃, λ, h̃)[:]) #if this is useful, can be combined with z_current
        Wsub = A + W * Diagonal(vcat(ones(Float32, M-P), z_diag[end-(P-1):end] .> 0))
        c̃, λ, h̃ = SumRep_sort_fully_cached(cache, A, W, h, z_diag, z_current, M, P)
        h̃ = Vector{Float32}(h̃)
        c̃_pwl, h̃_pwl = c̃[M-P+1:end,:], h̃[M-P+1:end]#@view(c̃[M-P+1:end,:]), @view(h̃[M-P+1:end])
        id_switch = dim_switch

        Δt_switch_local = Δt_switch #now use open interval for the next switching time
    

        ### Edge case:
        if tmax-ttotal < Δt_switch_local #if the next switching time is less than Δt, we cannot compute the next 
            #switching time. and have to assume that the system is in the last region
            t_local = t_points[idx_begin:end] .- ttotal
            z_local = real.(ContPLRNNSolution(t_local, c̃, λ, h̃))
            z_out = hcat(z_out, z_local)
            break
        end

        # Compute the next switching time
        iterations += 1
    end
    return save_switching_times ? (z_out, t_switch_vec, dim_switch_vec) : z_out
end