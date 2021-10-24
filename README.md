# variational_gaussian_process

## TL;DR

This Tensorflow-Probability model is for people who need a counterfactual, a deep-control, or a prediction of what we expected to happen in the absence of a control. This is acieved by selecting features using [cointegration](docs/EG_1987.pdf) and training a model using those features.

---

## Posterior predictive of a variational Gaussian process

This distribution implements the variational Gaussian process (VGP), as
described in [Titsias, 2009](docs/titsias09a.pdf) and [Hensman, 2013](docs/1309.6835.pdf). The VGP is an
inducing point-based approximation of an exact GP posterior
(see Mathematical Details, below). Ultimately, this Distribution class
represents a marginal distrbution over function values at a
collection of `index_points`. It is parameterized by

- a kernel function,
- a mean function,
- the (scalar) observation noise variance of the normal likelihood,
- a set of index points,
- a set of inducing index points, and
- the parameters of the (full-rank, Gaussian) variational posterior
  distribution over function values at the inducing points, conditional on
  some observations.

A VGP is "trained" by selecting any kernel parameters, the locations of the
inducing index points, and the variational parameters. [Titsias, 2009](docs/titsias09a.pdf) and
[Hensman, 2013](docs/1309.6835.pdf) describe a variational lower bound on the marginal log
likelihood of observed data, which this class offers through the
`variational_loss` method (this is the negative lower bound, for convenience
when plugging into a TF Optimizer's `minimize` function).
Training may be done in minibatches.

[Titsias, 2009](docs/titsias09a.pdf) describes a closed form for the optimal variational
parameters, in the case of sufficiently small observational data (ie,
small enough to fit in memory but big enough to warrant approximating the GP
posterior). A method to compute these optimal parameters in terms of the full
observational data set is provided as a staticmethod,
`optimal_variational_posterior`. It returns a
`MultivariateNormalLinearOperator` instance with optimal location and
scale parameters.

#### Mathematical Details

##### Notation

We will in general be concerned about three collections of index points, and
it'll be good to give them names:

- `x[1], ..., x[N]`: observation index points -- locations of our observed
  data.
- `z[1], ..., z[M]`: inducing index points -- locations of the
  "summarizing" inducing points
- `t[1], ..., t[P]`: predictive index points -- locations where we are
  making posterior predictions based on observations and the variational
  parameters.

To lighten notation, we'll use `X, Z, T` to denote the above collections.
Similarly, we'll denote by `f(X)` the collection of function values at each of
the `x[i]`, and by `Y`, the collection of (noisy) observed data at each `x[i]. We'll denote kernel matrices generated from pairs of index points as `K_tt`, `K_xt`, `K_tz`, etc, e.g.,

```none
         | k(t[1], z[1])    k(t[1], z[2])  ...  k(t[1], z[M]) |
  K_tz = | k(t[2], z[1])    k(t[2], z[2])  ...  k(t[2], z[M]) |
         |      ...              ...                 ...      |
         | k(t[P], z[1])    k(t[P], z[2])  ...  k(t[P], z[M]) |
```

##### Preliminaries

A Gaussian process is an indexed collection of random variables, any finite
collection of which are jointly Gaussian. Typically, the index set is some
finite-dimensional, real vector space, and indeed we make this assumption in
what follows. The GP may then be thought of as a distribution over functions
on the index set. Samples from the GP are functions _on the whole index set_;
these can't be represented in finite compute memory, so one typically works
with the marginals at a finite collection of index points. The properties of
the GP are entirely determined by its mean function `m` and covariance
function `k`. The generative process, assuming a mean-zero normal likelihood
with stddev `sigma`, is

```none
  f ~ GP(m, k)
  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
```

In finite terms (ie, marginalizing out all but a finite number of f(X)'sigma),
we can write

```none
  f(X) ~ MVN(loc=m(X), cov=K_xx)
  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
```

Posterior inference is possible in analytical closed form but becomes
intractible as data sizes get large. See [Rasmussen, 2006][3] for details.

##### The VGP

The VGP is an inducing point-based approximation of an exact GP posterior,
where two approximating assumptions have been made: 1. function values at non-inducing points are mutually independent
conditioned on function values at the inducing points, 2. the (expensive) posterior over function values at inducing points
conditional on observations is replaced with an arbitrary (learnable)
full-rank Gaussian distribution,
`none q(f(Z)) = MVN(loc=m, scale=S), `
where `m` and `S` are parameters to be chosen by optimizing an evidence
lower bound (ELBO).
The posterior predictive distribution becomes

```none
  q(f(T)) = integral df(Z) p(f(T) | f(Z)) q(f(Z))
          = MVN(loc = A @ m, scale = B^(1/2))
```

where

```none
  A = K_tz @ K_zz^-1
  B = K_tt - A @ (K_zz - S S^T) A^T
```

_The approximate posterior predictive distribution `q(f(T))` is what the
`VariationalGaussianProcess` class represents._

Model selection in this framework entails choosing the kernel parameters,
inducing point locations, and variational parameters. We do this by optimizing
a variational lower bound on the marginal log likelihood of observed data. The
lower bound takes the following form (see [Titsias, 2009](docs/titsias09a.pdf) and
[Hensman, 2013](docs/1309.6835.pdf) for details on the derivation):

```none
  L(Z, m, S, Y) = (
      MVN(loc=(K_zx @ K_zz^-1) @ m, scale_diag=sigma).log_prob(Y) -
      (Tr(K_xx - K_zx @ K_zz^-1 @ K_xz) +
       Tr(S @ S^T @ K_zz^1 @ K_zx @ K_xz @ K_zz^-1)) / (2 * sigma^2) -
      KL(q(f(Z)) || p(f(Z))))
```

where in the final KL term, `p(f(Z))` is the GP prior on inducing point
function values. This variational lower bound can be computed on minibatches
of the full data set `(X, Y)`. A method to compute the _negative_ variational
lower bound is implemented as `VariationalGaussianProcess.variational_loss`.

##### Optimal variational parameters

As described in [Titsias, 2009](docs/titsias09a.pdf), a closed form optimum for the variational
location and scale parameters, `m` and `S`, can be computed when the
observational data are not prohibitively voluminous. The
`optimal_variational_posterior` function to computes the optimal variational
posterior distribution over inducing point function values in terms of the GP
parameters (mean and kernel functions), inducing point locations, observation
index points, and observations. Note that the inducing index point locations
must still be optimized even when these parameters are known functions of the
inducing index points. The optimal parameters are computed as follows:

```none
  C = sigma^-2 (K_zz + K_zx @ K_xz)^-1
  optimal Gaussian covariance: K_zz @ C @ K_zz
  optimal Gaussian location: sigma^-2 K_zz @ C @ K_zx @ Y
```

The process is expecting your data to be structured as so;

- embedding: an indicator of a cohort of data (ie: embed_id_id)
- ds: a daily time series value (ie: business_date)
- y: a target kpi (ie: royalty_sales)

```
    |embedding  |ds        |y      |
    --------------------------------
    |qa_5       |2019-01-01|  -1.55|
    |qa_4       |2019-01-01|   0.41|
    |qa_0       |2019-01-01|   1.05|
```
