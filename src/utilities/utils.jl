using JSON
using JLD2: @save, @load
using Lux: Chain, Dense, glorot_uniform
using Flux: cpu, gpu #not known anymore in new version of Lux, but maybe can use Flux if needed

const RES = "Results"

"""
    TrainingRuntime

Mutable container for all training run state that is generated and updated
during a single run. Kept separate from experiment config (`args`) and the
diagnostic context (`TrainingContext`).

Fields:
- `save_path`: directory where run artifacts are written
- `epoch`:     current training epoch (1-indexed; incremented after each epoch)
- `losses`:    per-epoch loss history
- `times`:     per-epoch wall-clock time history (seconds)
- `N`:         observation dimension
"""
mutable struct TrainingRuntime
    save_path::String
    epoch::Int
    losses::Vector{Float32}
    times::Vector{Float32}
    N::Int
end

TrainingRuntime(save_path::String, n_epochs::Int, N::Int) =
    TrainingRuntime(save_path, 1, zeros(Float32, n_epochs), zeros(Float32, n_epochs), N)

"""
    TrainingContext

Mutable container for live, non-serializable runtime objects used exclusively
for crash diagnostics. Populated by `main_routine` and consumed by
`main_routine_safe` in the error-handling path. Never persisted.

Fields:
- `cont_model`: the Lux model struct
- `p`:          current parameter values
- `st`:         current model state
- `rt`:         the `TrainingRuntime` created for this run (available after
                folder setup in `main_routine`)
"""
mutable struct TrainingContext
    cont_model
    p
    st
    rt::Union{TrainingRuntime, Nothing}
    TrainingContext() = new(nothing, nothing, nothing, nothing)
end

"""
    create_folder_structure(exp::String, run::Int)

Creates basic saving structure for a single run/experiment.
"""
function create_folder_structure(exp::String, name::String, run::Int)::String
    # create folder
    path_to_run = joinpath(RES, exp, name, format_run_ID(run))
    mkpath(joinpath(path_to_run, "checkpoints"))
    mkpath(joinpath(path_to_run, "plots"))
    mkpath(joinpath(path_to_run, "plots/predictions"))
    mkpath(joinpath(path_to_run, "plots/generated"))
    return path_to_run
end

function format_run_ID(run::Int)::String
    # only allow three digit numbers
    @assert run < 1000
    return string(run, pad = 3)
end

"""
    store_hypers(args, path)

Persist experiment hyperparameters to `args.json` inside `path`.
"""
store_hypers(args::Dict, path::String) =
    open(joinpath(path, "args.json"), "w") do f
        JSON.print(f, args, 4)
    end

"""
    store_hypers(args, rt, path)

Persist experiment config combined with run-time summary to `args.json` inside
`path`. Runtime fields from `rt` (`save_path`, `epoch`, `N`, `losses`, `times`)
are merged with `args` so the output format is identical to the single-argument
overload for compatibility with downstream analysis scripts.
"""
function store_hypers(args::Dict, rt::TrainingRuntime, path::String)
    combined = merge(args, Dict(
        "save_path" => rt.save_path,
        "epoch"     => rt.epoch,
        "N"         => rt.N,
        "losses"    => rt.losses,
        "times"     => rt.times,
    ))
    open(joinpath(path, "args.json"), "w") do f
        JSON.print(f, combined, 4)
    end
end

function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end

load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))

load_json_f32(path) = convert_to_Float32(JSON.parsefile(path; dicttype=Dict{String,Any}))

save_model(p, st, path::String) = @save path p st

function check_for_NaNs(θ)
    nan = false
    for p in θ
        nan = nan || !isfinite(sum(p))
    end
    return nan
end

"""
    find_latest_model(run_path)

Search the folder given by `run_path` for the latest `model_[EPOCH].bson` and
return its path.
"""
function find_latest_model(run_path::String)::String
    files = filter(x -> endswith(x, ".jld2"), readdir(joinpath(run_path, "checkpoints")))
    n = length(files)
    ep_vec = Vector{Int}(undef, n)
    for i = 1:n
        ep_model = split(files[i], "_")[end]
        ep = parse(Int, split(ep_model, ".")[1])
        ep_vec[i] = ep
    end
    return joinpath(run_path, "checkpoints", files[argmax(ep_vec)])
end

replace_win_path(s::String) = replace(s, "\\" => "/")
