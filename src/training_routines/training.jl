using Lux,
    ComponentArrays,
    Optimisers,
    Random,
    Plots,
    LinearAlgebra,
    MKL,
    Zygote,
    JLD2

using Statistics: mean
using Printf

using Flux: mse

using ..Utilities
using ..ObservationModels: init_state


function compute_loss(X, t_batch, tfrec, ps, st; λ_reg::Float32 = 0.0f0)
    Z, st = tfrec((X, t_batch), ps, st)
    X̃ = tfrec.obs_model(Z, ps.obs_model, st.obs_model)
    
    # Main MSE loss
    mse_loss = mse(X̃[:, :, :], @view(X[:, :, 2:end]))
    
    # A matrix regularization: penalize values larger than 0
    A_reg = λ_reg * sum(max.(ps.cont_model.A+diag(ps.cont_model.W), 0.0f0))
    
    total_loss = mse_loss + A_reg
    return total_loss, X̃, ps, st
end



function training_callback(p, loss, X, Z, tfrec, st, rt::TrainingRuntime, args::AbstractDict)
    epoch = rt.epoch
    if epoch % args["saving_interval"] == 0
        model_filename = joinpath(rt.save_path, "checkpoints", "epoch_$(epoch).jld2")
        save_model(p, st, model_filename)
        if args["plot_and_store_predictions"] && X !== nothing && Z !== nothing
            fig = plot_predictions(X, Z; epoch = epoch)
            savefig(fig, joinpath(rt.save_path, "plots/predictions", "prediction_$(epoch).png"))
        end
    end
    return false
end

function generate_predictions(tfrec, D, p, st, rt::TrainingRuntime, args::AbstractDict)
    epoch = rt.epoch
    n = min(1000, size(D.X, 1))
    init_cond = init_state(tfrec.obs_model, D.X[1, :], p.obs_model, st.obs_model)
    t_points = D.tvec[1:n]
    cache = BatchCache{Any,Any}()
    Z_gen = compute_alrnn_trajectory_fully_cached(cache, t_points, p.cont_model.A, p.cont_model.W, p.cont_model.h, init_cond, tfrec.cont_model.P)
    X_gen = @views tfrec.obs_model(Z_gen, p.obs_model, st.obs_model)

    X_cpu = D.X[1:n, :, 1] |> cpu
    X_gen_cpu = permutedims(X_gen, (2, 1)) |> cpu

    fig = plot_predictions(X_cpu, X_gen_cpu)
    savefig(fig, joinpath(rt.save_path, "plots/generated", "generated_$(epoch).png"))
end


function training_progress(loss, time, rt::TrainingRuntime, args::AbstractDict)
    epoch = rt.epoch
    rt.losses[epoch] = loss
    rt.times[epoch] = time
    if epoch % args["saving_interval"] == 0
        npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
        npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
        label     = get(args, "worker_label", "")
        epoch_w   = length(string(args["epochs"]))
        epoch_str = lpad(string(epoch), epoch_w)
        mse_str   = @sprintf("%.4e", loss)
        time_str  = @sprintf("%7.2f", time)
        println("$(label)Epoch $(epoch_str) | MSE $(mse_str) | Time: $(time_str)s")
    end
    rt.epoch += 1
    return false
end

# Helper function to check if parameters contain NaN or Inf values
function any_parameters_nan(ps)
    return _check_parameters_recursive(ps)
end

# Recursive function to check any nested parameter structure
function _check_parameters_recursive(obj)
    if obj isa AbstractArray
        return any(x -> isnan(x) || isinf(x), obj)
    elseif obj isa NamedTuple || obj isa Dict
        for (key, value) in pairs(obj)
            if _check_parameters_recursive(value)
                return true
            end
        end
    elseif obj isa Number
        return isnan(obj) || isinf(obj)
    end
    return false
end

function _clip_gradients!(grads, threshold)
    corrupted = false
    for (key, value) in pairs(grads)
        if value isa AbstractArray
            grads[key] = clamp.(value, -threshold, threshold)
            if any(x -> isnan(x) || isinf(x), value)
                corrupted = true
            end
        elseif value isa NamedTuple || value isa Dict
            sub_corrupted = _clip_gradients!(value, threshold)
            corrupted = corrupted || sub_corrupted
        end
    end
    return corrupted
end

function clip_parameters!(ps, threshold)
    _clip_parameters_recursive!(ps, threshold)
end

# Recursive function to clip any nested parameter structure
function _clip_parameters_recursive!(obj, threshold)
    if obj isa AbstractArray
        obj .= clamp.(obj, -threshold, threshold)
    elseif obj isa NamedTuple
        for (key, value) in pairs(obj)
            _clip_parameters_recursive!(value, threshold)
        end
    elseif obj isa Dict
        for (key, value) in pairs(obj)
            _clip_parameters_recursive!(value, threshold)
        end
    elseif obj isa Number
        # For scalar numbers, we can't modify them in place, but this case is rare
        # in typical parameter structures
        nothing
    end
end


function loss_gradient(X, t_batch, tfrec, ps, st; λ_reg::Float32 = 1.0f-3)
    #compare compute_gradients.impl in LuxZygoteExt.training.jl
    (loss, ret...), back = pullback(p -> compute_loss(X, t_batch, tfrec, p, st; λ_reg=λ_reg), ps)
    gs = real.(back((one(loss), repeat([nothing], length(ret))...))[1])
    #it creates complex numbers due to the complex numbers in the model
    return gs, (loss, ret...)
end

function training_loop(tfrec, D, opt, ps, st, args::AbstractDict, rt::TrainingRuntime)
    ps = ComponentArray(ps)
    st_opt = Optimisers.setup(opt, ps)

    T, S = args["sequence_length"], args["batch_size"]
    σ_n = args["gaussian_noise_level"]
    Sₑ = get(args, "batches_per_epoch", 1)

    ηₛ = args["start_lr"]
    ηₑ = args["end_lr"]
    E = args["epochs"]

    γ = exp(log(ηₑ / ηₛ) / E)

    # Corrupted-batch tracking is internal to this loop; not stored in args or rt
    consecutive_corrupted_batches = 0
    total_corrupted_batches = 0
    corrupted_batch_times = Float64[]
    
    model_filename = joinpath(rt.save_path, "checkpoints", "epoch_0.jld2")
    save_model(ps, st, model_filename)


    for epoch = 1:E
        t₁ = time_ns()
        Optimisers.adjust!(st_opt, eta = ηₛ * γ^epoch)
        if @isdefined(clear_eigen_cache!)
            clear_eigen_cache!()
        end
        last_loss = 0.0f0
        X̃_last = nothing
        X̃_pred_last = nothing
        
        for sₑ = 1:Sₑ
            batch_t_start = time_ns()
            X̃, t_batch = sample_batch(D, T, S)

            σ_n > zero(σ_n) ? add_gaussian_noise!(X̃, σ_n) : nothing

            λ_reg = get(args, "A_matrix_regularization", 0.0f0)
            
            gs = nothing
            loss = 0.0f0
            X̃_pred = nothing
            
            try
                gs, (loss, X̃_pred, ps, st) = loss_gradient(X̃, t_batch, tfrec, ps, st; λ_reg=λ_reg)
            catch err
                if isa(err, ArgumentError) && occursin("matrix contains Infs or NaNs", string(err))
                    batch_time = (time_ns() - batch_t_start) / 1e9
                    @error "Numerical instability detected in forward pass, skipping this batch" epoch=epoch batch=sₑ error=err
                    consecutive_corrupted_batches += 1
                    total_corrupted_batches += 1
                    push!(corrupted_batch_times, batch_time)
                    if consecutive_corrupted_batches > 10
                        avg_cb_time = isempty(corrupted_batch_times) ? 0.0 : mean(corrupted_batch_times)
                        @error "Too many consecutive corrupted batches - stopping training" epoch=epoch
                        @error "Average corrupted batch time: $(round(avg_cb_time, digits=3))s"
                        
                        # Save corrupted model
                        corrupted_model_filename = joinpath(settings["save_path"], "checkpoints", "corrupted_model_epoch_$(e)_batch_$(sₑ).jld2")
                        @error "Saving corrupted model to: $corrupted_model_filename"
                        save_model(ps, st, corrupted_model_filename)
                        
                        # Save corrupted batch
                        corrupted_batch_filename = joinpath(settings["save_path"], "checkpoints", "corrupted_batch_epoch_$(e)_batch_$(sₑ).jld2")
                        @error "Saving corrupted batch to: $corrupted_batch_filename"
                        JLD2.@save corrupted_batch_filename X̃ t_batch epoch=e batch=sₑ
                        
                        # Also save training history
                        npzwrite(joinpath(settings["save_path"], "times.npy"), settings["times"])
                        npzwrite(joinpath(settings["save_path"], "losses.npy"), settings["losses"])
                        
                        break
                    end
                    continue
                else
                    rethrow(err)
                end
            end
            
            if sₑ == Sₑ
                last_loss = loss
                X̃_last = X̃
                X̃_pred_last = X̃_pred
            end

            if gs === nothing
                @error "No gradients computed, skipping this batch" epoch=epoch batch=sₑ
                consecutive_corrupted_batches += 1
                total_corrupted_batches += 1
                push!(corrupted_batch_times, 0.0)
                continue
            end
            
            if any_parameters_nan(ps)
                @error "Parameters contain NaN/Inf before gradient processing - MODEL CORRUPTED" epoch=epoch batch=sₑ
                save_model(ps, st, model_filename)
                npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
                npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
                return ps, st
            end
            
            clip_threshold = 1.0f3
            gradients_corrupted = _clip_gradients!(gs, clip_threshold)
           
            if gradients_corrupted
                batch_time = (time_ns() - batch_t_start) / 1e9
                consecutive_corrupted_batches += 1
                total_corrupted_batches += 1
                push!(corrupted_batch_times, 0.0)
                @error "Gradients still corrupted after clipping, skipping this batch" epoch=epoch batch=sₑ
                @error "Consecutive corrupted batches: $consecutive_corrupted_batches"
                @error "Total corrupted batches: $total_corrupted_batches"
                if consecutive_corrupted_batches > 10
                    avg_cb_time = isempty(corrupted_batch_times) ? 0.0 : mean(corrupted_batch_times)
                    @error "Too many consecutive corrupted batches - stopping training" epoch=epoch
                    @error "Average corrupted batch time: $(round(avg_cb_time, digits=3))s"

                    # Save corrupted model
                    corrupted_model_filename = joinpath(settings["save_path"], "checkpoints", "corrupted_model_epoch_$(e)_batch_$(sₑ).jld2")
                    @error "Saving corrupted model to: $corrupted_model_filename"
                    save_model(ps, st, corrupted_model_filename)
                    
                    # Save corrupted batch
                    corrupted_batch_filename = joinpath(settings["save_path"], "checkpoints", "corrupted_batch_epoch_$(e)_batch_$(sₑ).jld2")
                    @error "Saving corrupted batch to: $corrupted_batch_filename"
                    JLD2.@save corrupted_batch_filename X̃ t_batch epoch=e batch=sₑ
                    
                    # Also save training history
                    npzwrite(joinpath(settings["save_path"], "times.npy"), settings["times"])
                    npzwrite(joinpath(settings["save_path"], "losses.npy"), settings["losses"])
                    
                    break
                end
                continue
            else
                consecutive_corrupted_batches = 0
            end

            if any_parameters_nan(ps)
                @error "Parameters contain NaN/Inf before optimizer update - MODEL CORRUPTED" epoch=epoch batch=sₑ
                save_model(ps, st, model_filename)
                npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
                npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
                return ps, st
            end

            Optimisers.update!(st_opt, ps, gs)
            
            if any_parameters_nan(ps)
                @error "Optimizer update produced NaN/Inf parameters - MODEL CORRUPTED" epoch=epoch batch=sₑ
                save_model(ps, st, model_filename)
                npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
                npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
                return ps, st
            end
            
            clip_parameters!(ps, 10.0f0)

            if isnan(loss)
                @warn "Loss is NaN in batch $sₑ of epoch $epoch, stopping training"
                save_model(ps, st, model_filename)
                npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
                npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
                return ps, st
            end
        end
        
        t₂ = time_ns()
        elapsed = (t₂ - t₁) / 1e9
        
        training_callback(ps, last_loss, X̃_last, X̃_pred_last, tfrec, st, rt, args)
        training_progress(last_loss, elapsed, rt, args)

        if isnan(last_loss)
            @warn "Loss is NaN, stopping training"
            save_model(ps, st, model_filename)
            npzwrite(joinpath(rt.save_path, "times.npy"), rt.times)
            npzwrite(joinpath(rt.save_path, "losses.npy"), rt.losses)
            break
        end
    end

    if total_corrupted_batches > 0
        avg_corrupted_time = isempty(corrupted_batch_times) ? 0.0 : mean(corrupted_batch_times)
        @info "Training completed - Final corrupted batch statistics:"
        @info "  Total corrupted batches: $total_corrupted_batches"
        @info "  Average corrupted batch time: $(round(avg_corrupted_time, digits=3))s"
        @info "  Total time lost to corrupted batches: $(round(sum(corrupted_batch_times), digits=3))s"
        @info "  Percentage of batches corrupted: $(round(100 * total_corrupted_batches / (E * Sₑ), digits=2))%"
    else
        @info "Training completed - No corrupted batches detected!"
    end

    return ps, st
end

