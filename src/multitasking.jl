#using Lux, DiffEqFlux, LinearAlgebra, DifferentialEquations, OptimizationOptimisers, OptimizationOptimJL, Random, ComponentArrays, BSON, Plots, Flux: mse
using Distributed,
    Lux,
    LuxCore,
    Random,
    Plots,
    NPZ,
    LinearAlgebra,
    MKL,
    ComponentArrays,
    Statistics,
    JSON

using JLD2: load

mutable struct Argument
    name::String
    value::Any
    id_str::String

    function Argument(name::String, value)
        if value isa Vector
            error("Please add an identifier name \
                  for arguments that are subject to grid search!")
        end
        return new(name, value, "")
    end
    Argument(name::String, value, id_str::String) = new(name, value, id_str)
end

# type aliases
ArgVec = Vector{Argument}
ArgDict = Dict{String, Any}
TaskVec = Vector{ArgDict}

add_to_name(name::String, arg::Argument) = name * "-" * arg.id_str * "_" * string(arg.value)

function prepare_base_name(default_name::String, args::ArgVec)::String
    name = default_name
    # filter name argument if specified by user
    arg = filter(arg -> arg.name == "name", args)
    if !isempty(arg)
        name = arg[1].value
    end
    return name
end

function check_arguments(defaults::ArgDict, args::ArgVec)
    for arg in args
        # check if arg exists
        @assert haskey(defaults, arg.name) "Argument/Setting <$(arg.name)> does not exist."

        # cast to correct type
        arg.value = arg.value .|> typeof(defaults[arg.name])
    end
end

function prepare_tasks(defaults::ArgDict, args::ArgVec, n_runs::Int)
    # check if arguments passed actually exist in default settings
    check_arguments(defaults, args)

    # extract multitasking name
    name = prepare_base_name(defaults["name"], args)

    # split arguments into the ones that are subject to 
    # undergo grid search and the ones constant
    const_args = filter(arg -> !(arg.value isa Vector), args)
    gs_args = filter(arg -> arg.value isa Vector, args)

    # overwrite default args with const args
    baseline_args = copy(defaults)
    for arg in const_args
        baseline_args[arg.name] = arg.value
        name = isempty(arg.id_str) ? name : add_to_name(name, arg)
    end
    baseline_args["name"] = name

    # done here, if no gs is performed
    tasks = [baseline_args]
    if !isempty(gs_args)
        tasks = generate_grid_search_tasks(baseline_args, gs_args)
    end

    # add multiple runs per task
    tasks = add_runs_to_tasks(tasks, n_runs)

    # attach fixed-width worker labels for aligned console output
    make_worker_labels!(tasks, gs_args, n_runs)

    return tasks
end

function add_runs_to_tasks(tasks::TaskVec, n_runs::Int)
    tasks_w_runs = TaskVec()
    for task in tasks
        for r = 1:n_runs
            task_cp = copy(task)
            task_cp["run"] = r
            push!(tasks_w_runs, task_cp)
        end
    end
    @assert length(tasks_w_runs) == length(tasks) * n_runs
    return tasks_w_runs
end

function generate_grid_search_tasks(args::ArgDict, gs_args::ArgVec)
    # initialize with first gs variable
    tasks = TaskVec()
    init_arg = gs_args[1]
    add_values_to_task!(tasks, args, init_arg)

    # loop over other variables
    for arg in gs_args[2:end]
        new_tasks = copy(tasks)
        for task in tasks
            add_values_to_task!(new_tasks, task, arg)
        end
        # keep "mix terms"
        tasks = new_tasks[length(tasks)+1:end]
    end
    return tasks
end

function replace_arg(args::ArgDict, arg::Argument)
    args_cp = copy(args)
    args_cp[arg.name] = arg.value
    args_cp["name"] = add_to_name(args_cp["name"], arg)
    return args_cp
end

function add_values_to_task!(tasks::TaskVec, task::ArgDict, arg::Argument)
    for v in arg.value
        push!(tasks, replace_arg(task, Argument(arg.name, v, arg.id_str)))
    end
end

"""
    make_worker_labels!(tasks, gs_args, n_runs)

Attach a fixed-width `"worker_label"` string to every task dict so that
epoch-level output lines from different workers stay visually aligned.

Each `key=value` token is left-padded as a whole so any padding appears
before the key name rather than between `=` and the value.
Format example: `(P=5, M=10 | run 1) `.
"""
function make_worker_labels!(tasks::TaskVec, gs_args::ArgVec, n_runs::Int)
    # Width of the widest "KEY=value" token for each grid-search argument.
    part_widths = [length(gs_args[i].id_str) + 1 +
                   maximum(length(string(v)) for v in gs_args[i].value)
                   for i in eachindex(gs_args)]
    # Digit width for the run number (e.g. 2 when n_runs >= 10).
    run_digit_width = length(string(n_runs))

    for task in tasks
        # Left-align key=value tokens (rpad: space after value).
        # Right-align the run digit only (lpad: space before number, conventional).
        run_str = "run $(lpad(string(task["run"]), run_digit_width))"
        if isempty(gs_args)
            task["worker_label"] = "[$(run_str)] "
        else
            parts = join(
                [rpad("$(gs_args[i].id_str)=$(task[gs_args[i].name])", part_widths[i])
                 for i in eachindex(gs_args)],
                " "
            )
            task["worker_label"] = "[$(parts) | $(run_str)] "
        end
    end
end

"""
    main_routine(args, ctx)

Function executed by every worker process.

`ctx` is a `TrainingContext` allocated by the caller (`main_routine_safe`).
It is updated in-place with live model objects so the caller can use them
for crash diagnostics. `ctx` is never stored in `args` and is never persisted.
"""
function main_routine(args::AbstractDict, ctx::TrainingContext)
    # num threads
    n_threads = Threads.nthreads()
    BLAS.set_num_threads(args["BLAS_threads"])
    label = get(args, "worker_label", "")
    println(
        "$(label)Running on $n_threads Julia thread(s) [BLAS threads: $(BLAS.get_num_threads())]",
    )

    # get computing device
    device = get_device(args)

    # dataset
    D = load_dataset(args["path_to_data"], args["path_to_time"], args["sequence_time_delta"]; device = device)

    # init model
    cont_model = initialize_model(args)

    # observation_model
    O = initialize_observation_model(args, D)

    # tfrec
    tfrec = TFRecur(cont_model, O, args["teacher_forcing_interval"])

    # optimizer
    opt = initialize_optimizer(args)

    # create directories and runtime state container
    save_path = create_folder_structure(args["experiment"], args["name"], args["run"])
    rt = TrainingRuntime(save_path, args["epochs"], size(D.X, 2))
    ctx.rt = rt

    # store initial hypers (before training)
    store_hypers(args, rt, save_path)

    if !isempty(args["pretrained_path"])
        @info "Using pretrained model @ $(args["pretrained_path"])."
        dic = load(args["pretrained_path"])
        p_init, st_init = LuxCore.setup(Random.default_rng(), tfrec)
        p, st = dic["p"], dic["st"]
        @assert length(ComponentArray(p)) == length(ComponentArray(p_init)) "Pretrained model has different number of parameters!"
    else
        p, st = LuxCore.setup(Random.default_rng(), tfrec)
    end

    # Populate ctx for crash diagnostics — not stored in args
    ctx.cont_model = cont_model
    ctx.p = p
    ctx.st = st

    # train
    p, st = training_loop(tfrec, D, opt, p, st, args, rt)

    # Update ctx with final trained parameters for post-training diagnostics
    ctx.p = p
    ctx.st = st

    # plot trajectory (use first trial for init condition and time vector)
    n = min(10000, size(D.X, 1))
    init_cond = init_state(O, D.X[1, :, 1], p.obs_model, st.obs_model)
    t_points = D.tvec[1:n, 1]
    cache = BatchCache{Any,Any}()
    Z_gen = compute_alrnn_trajectory_fully_cached(cache, t_points, p.cont_model.A, p.cont_model.W, p.cont_model.h, init_cond, tfrec.cont_model.P)
    #Z_gen = cont_model((init_cond, t_points), p.cont_model, st.cont_model)[1]
    X_gen = O(Z_gen, p.obs_model, st.obs_model)
    npzwrite(joinpath(rt.save_path, "trajs.npy"), X_gen)
    plot_reconstruction(permutedims(X_gen, (2, 1)), D.X[1:n, :, 1], joinpath(rt.save_path, "final_generated.png"))
    npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
    npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
    store_hypers(args, rt, rt.save_path)
end

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int, args)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks with error handling
    results = pmap(main_routine_safe, tasks)
    
    # Process results and report errors
    successful_runs = 0
    failed_runs = 0
    
    for (i, result) in enumerate(results)
        if result isa Dict && haskey(result, "status") && result["status"] == "error"
            failed_runs += 1
            println("❌ Run $i failed: $(result["error"])")
        else
            successful_runs += 1
            println("✅ Run $i completed successfully")
        end
    end
    
    println("\n📊 Summary:")
    println("   Successful runs: $successful_runs")
    println("   Failed runs: $failed_runs")
    println("   Total runs: $(length(results))")

    # Release the accumulated results on the master to reclaim memory.
    results = nothing
    GC.gc()

    return nothing
end


"""
    compute_jacobian_plrnn(A, W, z)

Compute the Jacobian matrix for a continuous PLRNN model.
The Jacobian is: J = A + W * diag(z > 0)
"""
function compute_jacobian_plrnn(A, W, z)
    M = length(z)
    
    # Convert A to matrix if it's a vector (diagonal case)
    if A isa AbstractVector
        A_mat = Diagonal(A)
    else
        A_mat = A
    end
    
    # Compute the diagonal matrix D(z) = diag(z > 0)
    D = Diagonal(z .> 0)
    
    # Jacobian: J = A + W * D(z)
    J = A_mat + W * D
    
    return J
end

"""
    compute_jacobian_alrnn(A, W, z, P)

Compute the Jacobian matrix for a continuous ALRNN model.
The Jacobian is: J = A + W * diag([ones(M-P); z[end-(P-1):end] > 0])
"""
function compute_jacobian_alrnn(A, W, z, P)
    M = length(z)
    
    # Convert A to matrix if it's a vector (diagonal case)
    if A isa AbstractVector
        A_mat = Diagonal(A)
    else
        A_mat = A
    end
    
    # Compute the diagonal matrix D(z) = diag([ones(M-P); z[end-(P-1):end] > 0])
    D_diag = vcat(ones(Float32, M-P), z[end-(P-1):end] .> 0)
    D = Diagonal(D_diag)
    
    # Jacobian: J = A + W * D(z)
    J = A_mat + W * D
    
    return J
end

"""
    extract_model_info(cont_model, p, st, args)

Extract model parameters and compute eigenvalues for error reporting.
"""
function extract_model_info(cont_model, p, st, args)
    try
        # Extract basic model parameters
        model_info = Dict()
        
        if haskey(p, :cont_model)
            cont_params = p.cont_model
            model_info["A"] = Array(cont_params.A)
            model_info["W"] = Array(cont_params.W)
            model_info["h"] = Array(cont_params.h)
            
            # Get model dimensions
            M = size(cont_params.A, 1)
            P = get(args, "pwl_units", 0)
            
            # Convert A to matrix if it's a vector (diagonal case)
            if cont_params.A isa AbstractVector
                A_mat = Diagonal(cont_params.A)
            else
                A_mat = cont_params.A
            end
            
            # Generate a few random states to analyze Jacobian eigenvalues
            num_states = 100
            states = [randn(Float32, M) for _ in 1:num_states]
            all_eigenvalues = []
            
            for (i, z) in enumerate(states)
                # Compute the actual Jacobian based on model type
                if args["model"] == "contPLRNN"
                    J = compute_jacobian_plrnn(cont_params.A, cont_params.W, z)
                elseif args["model"] == "contALRNN"
                    J = compute_jacobian_alrnn(cont_params.A, cont_params.W, z, P)
                else
                    # Fallback for unknown model types
                    J = A_mat + cont_params.W
                end
                
                # Compute eigenvalues
                λ = eigvals(J)
                push!(all_eigenvalues, λ)
            end
            
            # Flatten all eigenvalues for analysis
            all_λ = vcat(all_eigenvalues...)
            model_info["eigenvalues"] = Array(all_λ)
            
            # Summary statistics
            model_info["max_real_part"] = maximum(real.(all_λ))
            model_info["min_real_part"] = minimum(real.(all_λ))
            model_info["max_magnitude"] = maximum(abs.(all_λ))
            model_info["min_magnitude"] = minimum(abs.(all_λ))
            model_info["stability_ratio"] = count(real.(all_λ) .< 0) / length(all_λ)
            #model_info["mean_real_part"] = mean(real.(all_λ))
            #model_info["mean_magnitude"] = mean(abs.(all_λ))
        end
        
        # Add model configuration
        model_info["model_type"] = args["model"]
        model_info["latent_dim"] = args["latent_dim"]
        if haskey(args, "pwl_units")
            model_info["pwl_units"] = args["pwl_units"]
        end
        
        return model_info
    catch extract_error
        return Dict("extraction_error" => string(extract_error))
    end
end

"""
    _find_latest_checkpoint(save_path)

Find the latest epoch checkpoint in the save_path/checkpoints directory.
Returns (path, p, st) or nothing if no checkpoint is found.
"""
function _find_latest_checkpoint(save_path)
    cp_dir = joinpath(save_path, "checkpoints")
    !isdir(cp_dir) && return nothing
    
    jld2s = filter(f -> endswith(f, ".jld2"), readdir(cp_dir))
    isempty(jld2s) && return nothing
    
    max_epoch = -1
    latest = ""
    for f in jld2s
        m = match(r"epoch_(\d+)\.jld2", f)
        m === nothing && continue
        ep = parse(Int, m.captures[1])
        if ep > max_epoch
            max_epoch = ep
            latest = f
        end
    end
    max_epoch < 0 && return nothing
    
    cp_path = joinpath(cp_dir, latest)
    try
        dic = load(cp_path)
        return (path=cp_path, p=dic["p"], st=dic["st"], epoch=max_epoch)
    catch
        return nothing
    end
end

function _complex_to_array(z)
    if z isa Complex
        return [real(z), imag(z)]
    elseif z isa AbstractArray
        return [_complex_to_array(zi) for zi in z]
    else
        return z
    end
end

"""
    main_routine_safe(args)

Safe wrapper around `main_routine` that catches errors and returns a structured
error result instead of re-throwing.

Allocates a `TrainingContext` locally and passes it explicitly to `main_routine`.
On failure, uses live model state from `ctx` for diagnostics. Falls back to the
latest on-disk checkpoint if `main_routine` failed before populating `ctx`.
"""
function main_routine_safe(args::AbstractDict)
    ctx = TrainingContext()

    try
        result = main_routine(args, ctx)
        return result
    catch e
        error_msg = "Error in main_routine for experiment $(args["experiment"])/$(args["name"])/$(args["run"]): $(e)"
        println(error_msg)
        
        save_path = ctx.rt !== nothing ? ctx.rt.save_path :
            create_folder_structure(args["experiment"], args["name"], args["run"])
        error_file = joinpath(save_path, "error_info.txt")

        # Get model state for diagnostics: prefer live state from ctx, fall back to disk
        model_info = Dict()
        diag_source = "none"
        try
            if ctx.p !== nothing && ctx.cont_model !== nothing
                model_info = extract_model_info(ctx.cont_model, ctx.p, ctx.st, args)
                diag_source = "live (epoch $(ctx.rt !== nothing ? ctx.rt.epoch : "?"))"
            else
                checkpoint = _find_latest_checkpoint(save_path)
                if checkpoint !== nothing
                    cont_model = initialize_model(args)
                    model_info = extract_model_info(cont_model, checkpoint.p, checkpoint.st, args)
                    diag_source = "checkpoint (epoch $(checkpoint.epoch))"
                end
            end
        catch extract_err
            model_info = Dict("extraction_error" => string(extract_err))
        end

        open(error_file, "w") do io
            println(io, "Error occurred during training:")
            println(io, "Error: ", e)
            println(io, "Stacktrace:")
            for (exc, bt) in Base.catch_stack()
                showerror(io, exc, bt)
                println(io)
            end
            
            if !isempty(model_info) && !haskey(model_info, "extraction_error")
                println(io, "\n" * "="^50)
                println(io, "MODEL PARAMETERS AND EIGENVALUES (source: $diag_source)")
                println(io, "="^50)
                
                println(io, "\nModel Configuration:")
                for key in ["model_type", "latent_dim", "pwl_units"]
                    haskey(model_info, key) && println(io, "  $key: $(model_info[key])")
                end
                
                if haskey(model_info, "A")
                    println(io, "\nParameter Statistics:")
                    println(io, "  A matrix - min: $(minimum(model_info["A"])), max: $(maximum(model_info["A"])), mean: $(mean(model_info["A"]))")
                    println(io, "  W matrix - min: $(minimum(model_info["W"])), max: $(maximum(model_info["W"])), mean: $(mean(model_info["W"]))")
                    println(io, "  h vector - min: $(minimum(model_info["h"])), max: $(maximum(model_info["h"])), mean: $(mean(model_info["h"]))")
                end
                
                if haskey(model_info, "eigenvalues")
                    println(io, "\nEigenvalue Analysis (Jacobian A + W*D(z)):")
                    println(io, "  Max real part: $(model_info["max_real_part"])")
                    println(io, "  Min real part: $(model_info["min_real_part"])")
                    println(io, "  Max magnitude: $(model_info["max_magnitude"])")
                    println(io, "  Min magnitude: $(model_info["min_magnitude"])")
                    println(io, "  Stability ratio: $(round(model_info["stability_ratio"] * 100, digits=1))%")
                    println(io, "  (Analyzed $(length(model_info["eigenvalues"]) ÷ model_info["latent_dim"]) random states)")
                end

                if haskey(model_info, "A")
                    params_file = joinpath(save_path, "model_parameters.json")
                    params_data = Dict(
                        "A" => model_info["A"],
                        "W" => model_info["W"], 
                        "h" => model_info["h"]
                    )
                    open(params_file, "w") do f
                        write(f, JSON.json(params_data))
                    end
                    
                    eigen_file = joinpath(save_path, "eigenvalues.json")
                    eigen_data = Dict(
                        "eigenvalues" => _complex_to_array(model_info["eigenvalues"]),
                        "summary_stats" => Dict(
                            "max_real_part" => model_info["max_real_part"],
                            "min_real_part" => model_info["min_real_part"],
                            "max_magnitude" => model_info["max_magnitude"],
                            "min_magnitude" => model_info["min_magnitude"],
                            "stability_ratio" => model_info["stability_ratio"]
                        )
                    )
                    open(eigen_file, "w") do f
                        write(f, JSON.json(eigen_data))
                    end
                    
                    println(io, "\nDetailed data saved to:")
                    println(io, "  - model_parameters.json")
                    println(io, "  - eigenvalues.json")
                end
            elseif haskey(model_info, "extraction_error")
                println(io, "\nModel info extraction failed: $(model_info["extraction_error"])")
            else
                println(io, "\nNo model state available for diagnostics (error occurred before initialization).")
            end
        end

        if ctx.rt !== nothing
            npzwrite(joinpath(save_path, "losses.npy"), ctx.rt.losses)
            npzwrite(joinpath(save_path, "times.npy"), ctx.rt.times)
        end

        return Dict(
            "status" => "error",
            "error" => string(e),
            "experiment" => args["experiment"],
            "name" => args["name"], 
            "run" => args["run"],
            "save_path" => save_path,
        )
    finally
        # Release references to the large training objects (model, parameters, state,
        # runtime) so Julia's GC can reclaim worker heap memory before the next task
        # is scheduled on this worker process.
        ctx.cont_model = nothing
        ctx.p          = nothing
        ctx.st         = nothing
        ctx.rt         = nothing
        GC.gc(true)
    end
end