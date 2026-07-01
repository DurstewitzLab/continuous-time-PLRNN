using Distributed
using ArgParse

function parse_ubermain()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--procs", "-p"
        help = "Number of parallel processes/workers to spawn."
        arg_type = Int
        default = 1

        "--runs", "-r"
        help = "Number of runs per experiment setting."
        arg_type = Int
        default = 5
    end
    return parse_args(s)
end

# parse number of procs, number of runs
ub_args = parse_ubermain()

# Pin MKL/BLAS threads before spawning workers so they inherit the constraint
ENV["MKL_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"

# start workers in SymbolicDSR env
addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using ContPLRNNTraining
@everywhere ENV["GKSwstype"] = "nul"

# Ensure workers are always removed — even when the script errors or is interrupted.
atexit(() -> begin
    wks = filter(w -> w != 1, workers())
    if !isempty(wks)
        @info "atexit: terminating $(length(wks)) worker process(es)…"
        try; rmprocs(wks; waitfor = 30); catch; end
    end
end)

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # list arguments here
    args = ContPLRNNTraining.ArgVec([
        Argument("experiment", "ALRNN_Lorenz63_Time"),
        Argument("name", "ALRNN_test"),
        Argument("model", "contALRNN"),
        Argument("epochs", 2000),
        Argument("pwl_units", [2, 5, 10], "P"),
        Argument("latent_dim", [20],"M"),
    ])

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
end

try
    ubermain(ub_args["runs"])
finally
    wks = filter(w -> w != 1, workers())
    if !isempty(wks)
        @info "Shutting down $(length(wks)) worker process(es)…"
        try; rmprocs(wks; waitfor = 60); catch; end
    end
end