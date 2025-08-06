# flambé
Fitness Lansdscape Architecture Models via Bayesian Estimation 

(or what happens when you apply a bit of **pyro** to your dish...)

## Introduction
Flyro is intended for fitting simple Bayesian probabilistic models to large mutational scanning datasets. For efficiency, it does so using stochastic variational inference (SVI) as implemented in the probabilistic programming package Pyro (https://pyro.ai/). It is intended to be both rigorously defined but very simple for non-experts to use.

The difference between flyro and other approaches is that it is fundamentally Bayesian - that is, rather than simply maximizing the likelihood of the data given a model, we attempt to assess the probabilities of various model parameters, which requires us to explicitly write out our priors. The advantage of this approach is that rather than inferring a single "best" set of parameters, you infer a probability distribution over possible parameters given the model design that allows you to more robustly assess error. Other than this, what **flambée** can do is quite similar to other tools for fitting simple regression models to mutational scanning datasets such as MoCHI, MaveNN, or RFA (and I'm sure many others). However, there is some functionality that I have explicitly added to flambee that those other models do not have.

## Basic usage
To fit a linear regression model, you can do the following:

```
import flambee as bf
X_train, y_train, y_err_train = make_tensors(X_train_df, ...)
linear_model = flyro.LinearModel(features)
linear_model.fit(X_train, y_train, y_err_train, 
                   num_particles=1, num_iterations=1500,
                   optimizer_type="ClippedAdam", lr=0.02, lrd=0.999)
y_pred = linear_model.predict(X_test)
```

## Priors
The three currently predefined models - linear, sigmoidal-independent, and sigmoidal couplings - all have a predetermined structure but the priors can be adjusted.

## Adding new models
A new model can be added by defining a new model and guide function:

```
def awesome_new_model(X, y=None, y_err=None, **prior_params):

  # establish priors
  indecipherable_squiggle = pyro.param("indecipherable_squiggle", dist.Normal(torch.tensor(is_loc, device=device), torch.tensor(is_scale, device=device))

  # math
  y_mean = f(X, indecipherable_squiggle)

  # deal with error however you'd like
  total_sigma = torch.sqrt(y_err**2 + a_pinch_of_salt)

  # observed data likelihood within a plate
  with pyro.plate("data", X.shape[0]):  
    pyro.sample("obs", dist.Normal(mean, total_sigma), obs=y)
```
