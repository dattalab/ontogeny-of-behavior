import jax
import optax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from toolz import dissoc
from statsmodels.gam.smooth_basis import BSplines


def piecewise_linear_bases(age, knots):
    h = jax.nn.relu((age[:, None] - knots[None, :])).T
    h = jnp.vstack([jnp.ones(age.shape), age, h])
    return h


def model_fun(params, bases, age_samples, true_age, counts, hypparams):
    concentrations = (
        jnp.exp(params["basis_weights"]) @ bases
    )  # shape (n_syllables, n_ages)

    concentrations = jnp.tile(concentrations.T[None], (len(counts), 1, 1))
    # shape (n_sessions, n_ages, n_syllables)

    if "mask" in hypparams:
        concentrations = concentrations[
            hypparams["mask"][0][..., None],
            jnp.arange(concentrations.shape[1])[None, :, None],
            hypparams["mask"][1][:, None, :],
        ]
        # print(concentrations.shape)

    # counts shape (n_sessions, n_syllables)
    if "mask" in hypparams:
        total_counts = counts[hypparams["mask"]].sum(axis=1)
    else:
        total_counts = counts.sum(axis=1)

    dir_mult = tfp.distributions.DirichletMultinomial(
        total_count=total_counts[:, None],
        concentration=concentrations,
        name="DirichletMultinomial",
    )

    if "mask" in hypparams:
        counts = counts[hypparams["mask"]]
    dir_probs = dir_mult.log_prob(counts[:, None, :])

    gauss_probs = tfp.distributions.Normal(
        loc=true_age[:, None], scale=hypparams["age_sd"]
    ).log_prob(age_samples[None])

    log_probs = dir_probs + gauss_probs

    return log_probs


def neg_log_likelihood(
    params, bases, age_samples, true_age, counts, hypparams, heldout=False
):
    # if heldout, use the heldout masks to compute likelihood
    if heldout and "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )

    log_probs = model_fun(params, bases, age_samples, true_age, counts, hypparams)

    # if heldout, switch mask values in hypparams dict back to original values
    if heldout and "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )

    sample_lls = jax.scipy.special.logsumexp(log_probs, axis=1)

    # since zmax == 1 here, I just have to subtract the number of samples
    sample_lls = sample_lls - log_probs.shape[1]

    total_lls = sample_lls.sum()
    if heldout:
        return -total_lls

    params_probs = tfp.distributions.Normal(
        loc=0, scale=hypparams["params_sd"]
    ).log_prob(params["basis_weights"])

    total_lls = total_lls + params_probs.sum()
    return -total_lls


def create_splines(x, df=6, degree=3, include_intercept=True):
    bs = BSplines(x, df=df, degree=degree, include_intercept=include_intercept)
    return bs, bs.basis.T


def age_normalizer_factory(min_age=0, max_age=150, log_transform=False):
    if log_transform:
        if min_age < 0:
            raise ValueError("Min age should be > 0")
        min_age = jnp.log(min_age + 1e-1)
        max_age = jnp.log(max_age)

    def age_normalizer(age):
        if log_transform:
            if jnp.sum(age < 0) > 0:
                raise ValueError("Ages should be > 0")
            age = jnp.log(jnp.where(age == 0, 1e-1, age))
        return (age - min_age) / (max_age - min_age)

    def age_unnormalizer(age):
        unnormalized_age = age * (max_age - min_age) + min_age
        if log_transform:
            unnormalized_age = jnp.exp(unnormalized_age)
        return unnormalized_age

    return age_normalizer, age_unnormalizer


def optimize(coef, loss_fn, lr=0.001, n_iter=1000):
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(coef)
    val_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(state, i):
        opt_state, params = state
        loss, grad = val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), (loss, grad)

    (opt_state, params), (loss, grad) = jax.lax.scan(
        step, (opt_state, coef), jnp.arange(n_iter)
    )

    return params, loss


def get_biological_age(
    params, bases, age_samples, chron_age, counts, hypparams, age_unnormalizer
):
    hypparams = dissoc(hypparams, "mask")
    log_probs = model_fun(params, bases, age_samples, chron_age, counts, hypparams)
    log_probs = jax.nn.softmax(log_probs, axis=1)
    predicted_bio_ages = age_samples[jnp.argmax(log_probs, axis=1)]
    return age_unnormalizer(predicted_bio_ages)


def model_fun_v2(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        jnp.exp(params["bio_basis_weights"]) @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        jnp.exp(params["size_basis_weights"]) @ bases["size"]
    )  # shape (n_syllables, n_sessions)

    bio_concentrations = jnp.tile(
        bio_concentrations.T[None], (size_concentrations.shape[1], 1, 1)
    )  # shape (n_sessions, n_ages, n_syllables)

    size_concentrations = jnp.tile(
        size_concentrations.T[:, None, :], (1, bio_concentrations.shape[1], 1)
    )  # shape (n_sessions, n_ages, n_syllables)

    concentrations = bio_concentrations + size_concentrations

    if "mask" in hypparams:
        concentrations = concentrations[
            hypparams["mask"][0][..., None],
            jnp.arange(concentrations.shape[1])[None, :, None],
            hypparams["mask"][1][:, None, :],
        ]

    if "mask" in hypparams:
        total_counts = counts[hypparams["mask"]].sum(axis=1)
    else:
        total_counts = counts.sum(axis=1)

    dir_mult = tfp.distributions.DirichletMultinomial(
        total_count=total_counts[:, None],
        concentration=concentrations,
        name="DirichletMultinomial",
    )

    if "mask" in hypparams:
        counts = counts[hypparams["mask"]]
    log_probs = dir_mult.log_prob(counts[:, None, :])

    gauss_probs = tfp.distributions.Normal(
        loc=true_age[:, None], scale=hypparams["age_sd"]
    ).log_prob(age_samples[None])

    log_probs = log_probs + gauss_probs

    return log_probs


def neg_log_likelihood_v2(
    params, bases, age_samples, true_age, counts, hypparams, heldout=False
):
    # if heldout, use the heldout masks to compute likelihood
    if heldout and "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )

    log_probs = model_fun_v2(params, bases, age_samples, true_age, counts, hypparams)

    # if heldout, switch mask values in hypparams dict back to original values
    if heldout and "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )

    sample_lls = jax.scipy.special.logsumexp(log_probs, axis=1)

    # since zmax == 1 here, I just have to subtract the number of samples
    sample_lls = sample_lls - log_probs.shape[1]

    total_lls = sample_lls.sum()
    if heldout:
        return -total_lls

    bio_dist = tfp.distributions.Normal(loc=0, scale=hypparams["bio_params_sd"])
    params_probs = bio_dist.log_prob(params["bio_basis_weights"])

    size_dist = tfp.distributions.Normal(loc=0, scale=hypparams["size_params_sd"])
    size_params_probs = size_dist.log_prob(params["size_basis_weights"])

    total_lls = total_lls + params_probs.sum() + size_params_probs.sum()
    return -total_lls
