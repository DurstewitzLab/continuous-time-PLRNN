module ContPLRNNTraining
using Reexport

@reexport using Lux

include("utilities/Utilities.jl")
@reexport using .Utilities

include("cont_utilities/ContUtilities.jl")
@reexport using .ContUtilities

include("models/ObservationModels.jl")
@reexport using .ObservationModels

include("models/Models.jl")
@reexport using .Models

include("training_routines/TrainingRoutines.jl")
@reexport using .TrainingRoutines

# meta stuff

include("parsing.jl")
export parse_commandline,
    parse_ubermain,
    initialize_model,
    initialize_optimizer,
    initialize_solver,
    get_device,
    argtable,
    initialize_observation_model

include("multitasking.jl")
export Argument,
    prepare_tasks,
    main_routine,
    main_routine_safe,
    ubermain,
    TrainingRuntime,
    load_dataset


end
