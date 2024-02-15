import jax
import optax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import tensorflow_probability.substrates.jax as tfp
from toolz import dissoc, partial
from statsmodels.gam.smooth_basis import BSplines


def create_masks(n_keep_sylls, n_syllables, n_sessions, rng):
    '''Create masks for heldout data. The mask is a tuple of two arrays, each corresponding
    to one index axis.'''
    mask = (
        np.arange(n_sessions)[:, None],
        np.array([rng.choice(n_syllables, size=n_keep_sylls, replace=False) for _ in range(n_sessions)]),
    )
    heldout_mask = (
        np.arange(n_sessions)[:, None],
        np.array([sorted(set(range(n_syllables)) - set(x)) for x in mask[1]])
    )
    return mask, heldout_mask


def piecewise_linear_bases(age, knots):
    h = jax.nn.relu((age[:, None] - knots[None, :])).T
    h = jnp.vstack([jnp.ones(age.shape), age, h])
    return h


def fit_params(counts, n_splines, age, age_samples, age_normalizer, n_syllables):
    spline_class, _ = create_splines(age_samples, df=n_splines)
    scale = 0.2

    A = spline_class.transform(age_normalizer(age)).T

    theta_list = []
    for i in range(n_syllables):
        _theta, _ = scipy.optimize.nnls(A.T, (counts + 1)[:, i] * scale)
        theta_list.append(_theta)
    theta_list = np.array(theta_list).T
    theta_list = np.where(theta_list == 0, 1e-3, theta_list)
    params = {"basis_weights": jnp.log(jnp.array(theta_list.T))}
    return params


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


def expected_syllable_frequencies(z, theta_age, B_age):
    bio_age = z

    def spline_interpolation(x, B):
        return jnp.interp(x, jnp.linspace(0, 1, 1000), B)

    spline_interpolation = jax.vmap(spline_interpolation, in_axes=(None, 0))

    B_at_age = spline_interpolation(bio_age, B_age)  # shape (K_age)
    bio_concentrations = jnp.dot(theta_age, B_at_age)  # shape (M)

    concentrations = bio_concentrations  # shape (M)

    # expected syllable frequencies
    expected_frequences = concentrations / jnp.sum(concentrations)

    return expected_frequences


def local_derivative(z, theta_age, B_age):
    '''Compute the local derivative of the model with respect to the animal's
    chronological age or biological age. Can also be normalized to syllable usage.'''
    fun = partial(
        expected_syllable_frequencies,
        theta_age=theta_age,
        B_age=B_age,
    )
    J = jax.jacfwd(fun)(z)
    exp_freqs = fun(z)

    # J has shape (M, )
    # J[i] = d expected_syllable_frequencies[i] / d bio_age

    fold_dependence = J / exp_freqs * z[None]
    # fold_dependence = J * z[None]
    return fold_dependence


def compute_local_derivative(ages, age_normalizer, params, n_splines):
    data = age_normalizer(ages)
    theta_age = params["basis_weights"]
    age_samples = np.linspace(0, 1, 1000)
    _, B_age = create_splines(age_samples, df=n_splines)

    fold_fun = partial(local_derivative, theta_age=jnp.exp(theta_age), B_age=B_age)
    return jax.vmap(fold_fun, in_axes=(0, ))(data)


### ~~~~~~~~~~~ version 2 of the model - add size basis ~~~~~~~~~~~ ###


def fit_params_v2(
    counts, n_splines, age, n_size_splines, n_syllables, age_samples, age_normalizer
):
    spline_class, splines = create_splines(age_samples, df=n_splines)
    scale = 0.2

    A = spline_class.transform(age_normalizer(age)).T

    theta_list = []
    for i in range(n_syllables):
        _theta, _ = scipy.optimize.nnls(A.T, (counts + 1)[:, i] * scale)
        theta_list.append(_theta)
    theta_list = np.array(theta_list).T
    theta_list = np.where(theta_list == 0, 1e-3, theta_list)
    params = {
        "bio_basis_weights": jnp.log(jnp.array(theta_list.T)),
        "size_basis_weights": jnp.zeros((n_syllables, n_size_splines)),
    }
    return params


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


def get_biological_age_v2(
    params, bases, age_samples, chron_age, counts, hypparams, age_unnormalizer
):
    hypparams = dissoc(hypparams, "mask")
    log_probs = model_fun_v2(params, bases, age_samples, chron_age, counts, hypparams)
    log_probs = jax.nn.softmax(log_probs, axis=1)
    predicted_bio_ages = age_samples[jnp.argmax(log_probs, axis=1)]
    return age_unnormalizer(predicted_bio_ages)


def expected_syllable_frequencies_v2(z, theta_age, theta_size, B_age, B_size):
    bio_age, size = z

    def spline_interpolation(x, B):
        return jnp.interp(x, jnp.linspace(0, 1, 1000), B)

    spline_interpolation = jax.vmap(spline_interpolation, in_axes=(None, 0))

    B_at_age = spline_interpolation(bio_age, B_age)  # shape (K_age)
    bio_concentrations = jnp.dot(theta_age, B_at_age)  # shape (M)

    B_at_size = spline_interpolation(size, B_size)  # shape (K_size)
    size_concentrations = jnp.dot(theta_size, B_at_size)  # shape (M)

    concentrations = bio_concentrations + size_concentrations  # shape (M)

    # expected syllable frequencies
    expected_frequences = concentrations / jnp.sum(concentrations)

    return expected_frequences


def local_derivative_v2(z, theta_age, theta_size, B_age, B_size):
    fun = partial(
        expected_syllable_frequencies_v2,
        theta_age=theta_age,
        theta_size=theta_size,
        B_age=B_age,
        B_size=B_size,
    )
    J = jax.jacfwd(fun)(z)
    exp_freqs = fun(z)

    # J has shape (M, 2)
    # J[i,0] = d expected_syllable_frequencies[i] / d bio_age
    # J[i,1] = d expected_syllable_frequencies[i] / d size

    fold_dependence = J / exp_freqs[:, None] * z[None]
    # fold_dependence = J * z[None]
    return fold_dependence


def compute_local_derivative_v2(ages, sizes, age_normalizer, params, n_splines, n_size_splines):
    data = jnp.stack([age_normalizer(ages), sizes], axis=1)
    theta_age = params["bio_basis_weights"]
    theta_size = params["size_basis_weights"]

    age_samples = np.linspace(0, 1, 1000)
    _, B_age = create_splines(age_samples, df=n_splines)
    _, B_size = create_splines(age_samples, df=n_size_splines)

    fold_fun = partial(local_derivative_v2, theta_age=jnp.exp(theta_age), theta_size=jnp.exp(theta_size), B_age=B_age, B_size=B_size)
    return jax.vmap(fold_fun, in_axes=(0, ))(data)


### ~~~~~~~~~~~ version 3 of the model - add size basis and individuality terms ~~~~~~~~~~~ ###

def model_fun_v3(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        jnp.exp(params["bio_basis_weights"]) @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        jnp.exp(params["size_basis_weights"]) @ bases["size"]
    )  # shape (n_syllables, n_sessions)

    individual_biases = jnp.exp(params["individual_biases"]) @ bases["individual"]  # shape (n_syllables, n_sessions)

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[:, None, :]  # shape (n_sessions, 1, n_syllables)
    # bio_concentrations = jnp.tile(
    #     bio_concentrations.T[None], (size_concentrations.shape[1], 1, 1)
    # )  # shape (n_sessions, n_ages, n_syllables)

    # size_concentrations = jnp.tile(
    #     size_concentrations.T[:, None, :], (1, bio_concentrations.shape[1], 1)
    # )  # shape (n_sessions, n_ages, n_syllables)

    individual_biases = individual_biases.T[:, None, :]  # shape (n_sessions, 1, n_syllables)

    concentrations = bio_concentrations + size_concentrations + individual_biases

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


def neg_log_likelihood_v3(
    params, bases, age_samples, true_age, counts, hypparams, heldout=False
):
    # if heldout, use the heldout masks to compute likelihood
    if heldout and "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )

    log_probs = model_fun_v3(params, bases, age_samples, true_age, counts, hypparams)

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


def fit_params_v3(
    counts, n_splines, age, n_size_splines, n_syllables, n_animals, age_samples, age_normalizer
):
    spline_class, splines = create_splines(age_samples, df=n_splines)
    scale = 0.2

    A = spline_class.transform(age_normalizer(age)).T

    theta_list = []
    for i in range(n_syllables):
        _theta, _ = scipy.optimize.nnls(A.T, (counts + 1)[:, i] * scale)
        theta_list.append(_theta)
    theta_list = np.array(theta_list).T
    theta_list = np.where(theta_list == 0, 1e-3, theta_list)
    params = {
        "bio_basis_weights": jnp.log(jnp.array(theta_list.T)),
        "size_basis_weights": jnp.zeros((n_syllables, n_size_splines)),
        "individual_biases": jnp.zeros((n_syllables, n_animals)),
    }
    return params
