

# Continuous-Time Piecewise-Linear Recurrent Neural Networks [ICML 2026 Poster]

# Introduction
This repository provides an implementation of the Continuous-Time Piecewise-Linear RNN (cPLRNN) used for dynamical systems (DS) reconstruction. 

The cPLRNNs are trained using backpropagation through time with a sparse teacher forcing protocol.


# 1. Julia implementation
To install the package, clone the repostiory and `cd` into the project folder:
Install the package in a new Julia environment:
```
julia> ]
(@v1.11) pkg> activate .
(ContPLRNNTraining) pkg> instantiate
```

## Running the Code
### Single Runs
To start a single training, execute the `main.jl` file, where arguments can be passed via command line. For example, to train a cPLRNN with 20 latent dimensions and 2 PWL units for 2000 epochs using 4 threads, while keeping all other training parameters at their default setting, call
```
$ julia -t4 --project main.jl --model contALRNN --latent_dim 20 --P 2 --epochs 2000
```
in your terminal of choice (bash/cmd). The [default settings](settings/defaults.json) can also be adjusted directly; one can then omit passing any arguments at the call site. The arguments are also listed in  in the [`argtable()`](src/parsing.jl) function.

### Multiple Runs + Grid Search
To run multiple trainings in parallel e.g. when grid searching hyperparameters, the `ubermain.jl` file is used. Currently, one has to adjust arguments which are supposed to differ from the [default settings](settings/defaults.json), and arguments that are supposed to be grid searched, in the `ubermain` function itself. This is as simple as adding an `Argument` to the `ArgVec` vector, which is passed the hyperparameter name (e.g. `latent_dim`), the desired value, and and identifier for discernibility and documentation purposes. If value is a vector of values, grid search for these hyperparameters is triggered. 
```Julia
function ubermain(n_runs::Int)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # list arguments here
    args = ContPLRNNTraining.ArgVec([
        Argument("experiment", "ALRNN_Lorenz63"),
        Argument("model", "contALRNN"),
        Argument("pwl_units", [0,1,2,3,4,5], "P"),
    ])

    [...]
end
```
This will run a grid search over `pwl_units` corresponding to the number of PWL units using the `contALRNN`.

The identifier (e.g. `"P"` in the snippet above) is only mandatory for arguments subject to grid search. Once Arguments are specified, call the ubermain file with the desired number of parallel worker proccesses (+ amount of threads per worker) and the number of runs per task/setting, e.g.
```{.sh}
$ julia -t2 --project ubermain.jl -p 20 -r 10
```
will queue 10 runs for each setting and use 20 parallel workers with each 2 threads.

## Specifics

### Model Architecture
Latent/Dynamics model
- contALRNN &rarr; [`contALRNN`](src/models/cont_alrnn.jl), where `pwl_units` controls the number of PWL units
with 
- Identity mapping &rarr; [`Identity`](src/models/identity.jl), to generate observations

### Data Format
Data for the algorithm is expected to be a single trajectory in form of a $T \times N$ matrix (file format: `.npy`), where $T$ is the total number of time steps and $N$ is the data dimensionality. [Examples](example_data/) are provided. If the time points the trajectory belongs to are not equally sampled, you can give the path to the $T$-dimensional vector (file format: `.npy`) via the `path_to_time`. If you just hand it `""`, it will assume equally distant points with a time difference `sequence_time_delta`.

### Training method
The cPLRNN is trained by backpropagation through time using sparse teacher forcing. The forcing interval is controlled by `teacher_forcing_interval`, which specifies the intervals at which the latent state is forced according to the observations in order to prevent exploding/vanishing gradients.

### Versions
- Julia 1.11.6
- Lux 0.16.10

# Citation
If you find the repository and/or paper helpful for your own research, please cite [our work](https://openreview.net/forum?id=JuaulCZ7gE).
```
@inproceedings{
braendle2026continuoustime,
title={Continuous-Time Piecewise-Linear Recurrent Neural Networks},
author={Alena Br{\"a}ndle and Lukas Eisenmann and Florian G{\"o}tz and Daniel Durstewitz},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=JuaulCZ7gE}
}
```

# Acknowledgements
This work was supported by the German Research Foundation (DFG) through the TRR 265 (subproject A06), individual grant Du 354/15-1 (project no. 502196519), and Du 354/14-1 (project no. 437610067) to DD within the FOR-5159.


## Settings for Rootfinder
The code uses a Interval Newton root finder that is custom designed to work for the functional form of the cPLNN solution

$$
z^{(i)}(t) = \sum_{l} \tilde{c}^{(i)}_l e^{\lambda^{(l)} t} + \tilde{h}^{(i)} \colon f^{(i)}(t; \mathbf{z}_0)
$$

It comes with several settings, that can be chosen but also have a default value.

Since we are doing a branch-and-prune search that in principle could do infinite iterations in case of ill-behaved functions (e.g. highly oscillatory), we added `max_iterations` to steer the maximum number of iterations.

In the [Interval Newton implementation](https://juliaintervals.github.io/IntervalRootFinding.jl/stable/) used as a basis, the interval is closed to halfed in each branch step. If we are looking at a rather big initial interval, one might have to do a lot of these branching steps. If we have a guess were the first root might be, we can subdivide the initial big interval (a,b) into two (a, a+`max_region_width`), (a+ `max_region_width`, b) and investigate the first interval. 

The Interval Newton method can find intervals in which exactly one root is enclosed. For our code, this is not enough; we need a good estimate of the actual position of the root $t_\text{root}$. For that, the function `refine` is for. It shrinks the found interval further

While $b-a$ > epsilon_zero
-  If we have a bracketing interval $f(a)\cdot f(b)<0$, we use the `find_zero` function form the `Roots.jl` package. 
-  Elseif we test the supremum of the interval $|f(b)|<$`delta_zero` &rarr; $t_\text{root} = b$
-  Elseif we test the infimum of the interval $|f(a)|<$`delta_zero` &rarr; $t_\text{root} = a$
-  Else refine the interval with another Interval Newton step.

Another potential numerical problem is that the switching time $t_\text{switch}$ returned by the root finder might not yield exactly $z^{(i_\text{switch})}(t_\text{switch})=0$, but values that slightly deviate in either direction, placing the new state slightly before or after the root.

If the state is slightly before the actual boundary crossing, the same root might be rediscovered, causing an infinite loop.<br>
**Solution:** Search for the root in the interval ($dt_{\text{switch}}$, $t_{\text{max}} - t_{\text{total}}$) instead of using $0$ as the lower bound.

Even if the value of $z^{(i_{\text{switch}})}(t_{\text{switch}})$ is exactly $0$, the matrix $\mathbf{D}$ is not initialized correctly when the boundary is crossed from negative to positive, because $d^{(i)}(t) = 0$ for both $z^i(t) < 0$ and $z^{(i)}(t) = 0$.<br>
**Solution:** Use $z_{\text{diag}} = z(t_{\text{switch}} + dt_{\text{diag}})$ to determine the diagonal vector $\mathbf{d}$.

The derivative of the switching time with respect to the parameters $\boldsymbol{\phi}$ is given by

$$
\frac{\partial t_\text{switch}}{\partial \boldsymbol{\phi}}
= -\frac{\frac{\partial f^{(i)}}{\partial \boldsymbol{\phi}}(t_\text{switch}, \boldsymbol{\phi})}
        {\frac{\partial f^{(i)}}{\partial t}(t_\text{switch}, \boldsymbol{\phi})}.
$$

If $|\frac{\partial f^{(i)}}{\partial t}(t_\text{switch}, \boldsymbol{\phi})|$ < `dz_tangent_threshold`, we discard the derivative. 
