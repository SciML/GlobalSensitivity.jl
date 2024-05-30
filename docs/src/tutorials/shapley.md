# Using the Shapley method in case of correlated inputs

One of the primary drawbacks of typical global sensitivity analysis methods is their
inability to handle correlated inputs. The Shapley method is one of the few methods
that can handle correlated inputs. The Shapley method is a game-theoretic approach
that is based on the idea of marginal contributions of each input to the output.

It has gained extensive popularity in the field of machine learning and is used to
explain the predictions of black-box models. Here we will use the Shapley method
on a Scientific Machine Learning (SciML) model to understand the impact of each
parameter on the output.

We will use a Neural ODE trained on a simulated dataset from the Spiral ODE model.
The Neural ODE is trained to predict output at a given time. The Neural ODE is
trained using the [SciML ecosystem](https://sciml.ai/).

As the first step let's generate the dataset.

```@example shapley
using GlobalSensitivity, OrdinaryDiffEq, Flux, SciMLSensitivity, LinearAlgebra
using Optimization, OptimizationOptimisers, Distributions, Copulas, CairoMakie

u0 = [2.0f0; 0.0f0]
datasize = 30
tspan = (0.0f0, 1.5f0)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1f0 2.0f0; -2.0f0 -0.1f0]
    du .= ((u .^ 3)'true_A)'
end
t = range(tspan[1], tspan[2], length = datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat = t))
```

Now we will define our Neural Network for the dynamics of the system. We will use
a 2-layer neural network with 10 hidden units in the first layer and the second layer.
We will use the `Chain` function from `Flux` to define our NN. A detailed tutorial on
is available [here](https://docs.sciml.ai/SciMLSensitivity/stable/examples/neural_ode/neural_ode_flux/).

```@example shapley
dudt2 = Flux.Chain(x -> x .^ 3,
    Flux.Dense(2, 10, tanh),
    Flux.Dense(10, 2))
p, re = Flux.destructure(dudt2) # use this p as the initial condition!
dudt(u, p, t) = re(p)(u) # need to restrcture for backprop!
prob = ODEProblem(dudt, u0, tspan)

θ = [u0; p] # the parameter vector to optimize

function predict_n_ode(θ)
    Array(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:end], saveat = t))
end

function loss_n_ode(θ)
    pred = predict_n_ode(θ)
    loss = sum(abs2, ode_data .- pred)
    loss
end

loss_n_ode(θ)

callback = function (state, l) #callback function to observe training
    display(l)
    return false
end

# Display the ODE with the initial parameter values.
callback(θ, loss_n_ode(θ))

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((p, _) -> loss_n_ode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 300)
```

Now we will use the Shapley method to understand the impact of each parameter on the
resultant of the cost function. We will use the `Shapley` function from `GlobalSensitivity`
to compute the so called Shapley Effects. We will first have to define some distributions
for the parameters. We will use the standard `Normal` distribution for all the parameters.

First let's assume no correlation between the parameters. Hence the covariance matrix
is passed as the identity matrix.

```@example shapley
d = length(θ)
mu = zeros(Float32, d)
#covariance matrix for the copula
Covmat = Matrix(1.0f0 * I, d, d)
#the marginal distributions for each parameter
marginals = [Normal(mu[i]) for i in 1:d]

copula = GaussianCopula(Covmat)
input_distribution = SklarDist(copula, marginals)

function batched_loss_n_ode(θ)
    # The copula returns samples of `Float64`s
    θ = convert(AbstractArray{Float32}, θ)
    prob_func(prob, i, repeat) = remake(prob; u0 = θ[1:2, i], p = θ[3:end, i])
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    sol = solve(
        ensemble_prob, Tsit5(), EnsembleThreads(); saveat = t, trajectories = size(θ, 2))
    out = zeros(size(θ, 2))
    for i in 1:size(θ, 2)
        out[i] = sum(abs2, ode_data .- sol[i])
    end
    return out
end

shapley_effects = gsa(
    batched_loss_n_ode, Shapley(; n_perms = 100, n_var = 100, n_outer = 10),
    input_distribution, batch = true)
```

```@example shapley
barplot(
    1:54, shapley_effects.shapley_effects;
    color = :green,
    figure = (; size = (600, 400)),
    axis = (;
        xlabel = "parameters",
        xticklabelrotation = 1,
        xticks = (1:54, ["θ$i" for i in 1:54]),
        ylabel = "Shapley Indices",
        limits = (nothing, (0.0, 0.2))
    )
)
```

Now let's assume some correlation between the parameters. We will use a correlation of 0.09 between
all the parameters.

```@example shapley
Corrmat = fill(0.09f0, d, d)
for i in 1:d
    Corrmat[i, i] = 1.0f0
end

#since the marginals are standard normal the covariance matrix and correlation matrix are the same
copula = GaussianCopula(Corrmat)
input_distribution = SklarDist(copula, marginals)
shapley_effects = gsa(
    batched_loss_n_ode, Shapley(; n_perms = 100, n_var = 100, n_outer = 100),
    input_distribution, batch = true)
```

```@example shapley
barplot(
    1:54, shapley_effects.shapley_effects;
    color = :green,
    figure = (; size = (600, 400)),
    axis = (;
        xlabel = "parameters",
        xticklabelrotation = 1,
        xticks = (1:54, ["θ$i" for i in 1:54]),
        ylabel = "Shapley Indices",
        limits = (nothing, (0.0, 0.2))
    )
)
```
