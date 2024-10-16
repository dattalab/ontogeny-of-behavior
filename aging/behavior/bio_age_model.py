import jax
import optax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
from copy import deepcopy
from tqdm.auto import tqdm
from dataclasses import dataclass
from sklearn.metrics import r2_score
from typing import Callable, Optional, Tuple, List
from toolz import dissoc, partial, valmap, merge, keyfilter
from statsmodels.gam.smooth_basis import UnivariateBSplines
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


@dataclass
class InitComponents:
    bases: dict
    age_samples: np.ndarray
    true_age: np.ndarray
    age_normalizer: Callable
    age_unnormalizer: Callable
    init: Callable
    params: dict
    model_version: int


def blacklist(d: dict, keys: list):
    """Remove keys from dictionary in `keys` list."""

    return keyfilter(lambda k: k not in keys, d)


# from @calebweinreb
def center_embedding(n: int):
    """Generate an orthonormal matrix that embeds R^(n-1) into the space of 0-sum vectors in R^n."""
    X = jnp.tril(jnp.ones((n, n)), k=-1)[1:]
    X = jnp.eye(n)[1:] - X / X.sum(1)[:, None]
    X = X / jnp.sqrt((X**2).sum(1))[:, None]
    return X.T


# from @calebweinreb
def raise_dim(arr, axis=0):
    """Raise dimension in specified axis by embedding into the space of 0-sum vectors."""
    arr = jnp.moveaxis(arr, axis, 0)
    k, *shape = arr.shape
    arr = arr.reshape(k, -1)
    arr = center_embedding(k + 1) @ arr
    arr = arr.reshape(k + 1, *shape)
    arr = jnp.moveaxis(arr, 0, axis)
    return arr


def scale_sd(sd, mu):
    """Scale a log-normal standard deviation by its mean so that the sd is equal when
    log-transformed data is re-exponentiated."""
    return jnp.sqrt(jnp.log(1 + jnp.square(sd / mu)))


def optimize(coef, loss_fn, lr=0.001, n_iter=1000) -> Tuple[dict, List[float]]:
    """Sets up and runs adam optimizer for setting model parameters.
    NOTE: it might make more sense to use SGD, as I recently learned that adam
    creates "privileged bases":
    https://www.lesswrong.com/posts/yrhu6MeFddnGRSLtQ/adam-optimizer-causes-privileged-basis-in-transformer-lm
    """
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


def create_masks(n_keep_sylls, n_syllables, n_sessions, rng):
    """Create masks for heldout data. The mask is a tuple of two arrays, each corresponding
    to one index axis."""
    mask = (
        np.arange(n_sessions)[:, None],
        np.array(
            [
                sorted(rng.choice(n_syllables, size=n_keep_sylls, replace=False))
                for _ in range(n_sessions)
            ]
        ),
    )
    heldout_mask = (
        np.arange(n_sessions)[:, None],
        np.array([sorted(set(range(n_syllables)) - set(x)) for x in mask[1]]),
    )
    return mask, heldout_mask


# old basis function - unused but could be useful?
def piecewise_linear_bases(age, knots):
    h = jax.nn.relu((age[:, None] - knots[None, :])).T
    h = jnp.vstack([jnp.ones(age.shape), age, h])
    return h


def model_setup(features: dict, hypparams: dict, model_version: int) -> InitComponents:

    bases, age_samples, age_normalizer, age_unnormalizer = _initialize_age_basis(
        hypparams
    )
    true_age = age_normalizer(features["ages"])

    if model_version > 1:
        bases["size"] = features["sizes"][None]

    init, bases = _initialize_parameters(model_version, bases, features, hypparams)
    params = init(
        features["counts"],
        features["ages"],
        age_samples,
        age_normalizer,
        hypparams["n_syllables"],
        n_splines=hypparams.get("n_splines"),
    )
    return InitComponents(
        bases,
        age_samples,
        true_age,
        age_normalizer,
        age_unnormalizer,
        init,
        params,
        model_version,
    )


def _initialize_parameters(model_version, bases, features, hypparams) -> Callable:
    if model_version == 0:
        init = partial(initialize_params, linear_model=True)
    elif model_version == 2:
        init = partial(
            initialize_params,
        )
    elif model_version == 3:
        init = partial(
            initialize_params,
            n_animals=hypparams["n_animals"],
        )
        bases["individual"] = jnp.array(features["mice"])
    elif model_version == 4:
        init = partial(
            initialize_params,
            n_animals=hypparams["n_animals"],
            model_development=True,
        )
        bases["individual"] = jnp.array(features["mice"])
    elif model_version == 5:
        init = partial(
            initialize_params,
            n_animals=hypparams["n_animals"],
            n_development_splines=hypparams["n_development_splines"],
            zero_mean=hypparams.get("zero_mean", True),
        )
        bases["individual"] = jnp.array(features["mice"])
        _, B_dev = create_splines(
            np.linspace(0, 1, hypparams["n_age_samples"]),
            df=hypparams["n_development_splines"],
        )
        bases["development"] = B_dev
    elif model_version == 6:
        init = partial(
            initialize_params,
            n_animals_per_sex=hypparams["n_animals_per_sex"],
            n_sexes=hypparams["n_sexes"],
            n_development_splines=hypparams["n_development_splines"],
            n_sex_splines=hypparams["n_sex_splines"],
        )
        bases["individual"] = jnp.array(features["mice"])
        bases["sex_id"] = jnp.array(features["sex"])
        _, B_dev = create_splines(
            np.linspace(0, 1, hypparams["n_age_samples"]),
            df=hypparams["n_development_splines"],
        )
        bases["development"] = B_dev
        _, B_sex = create_splines(
            np.linspace(0, 1, hypparams["n_age_samples"]), df=hypparams["n_sex_splines"]
        )
        bases["sex"] = B_sex
    else:
        init = initialize_params
    return init, bases


def _initialize_age_basis(hypparams):

    age_normalizer, age_unnormalizer = age_normalizer_factory(
        min_age=hypparams["min_age"],
        max_age=hypparams["max_age"],
        log_transform=hypparams.get("log_age", False),
    )
    age_samples = np.linspace(0, 1, hypparams["n_age_samples"])

    if hypparams.get("n_splines") is None:
        return {}, age_samples, age_normalizer, age_unnormalizer

    if "extra_knots" in hypparams and not isinstance(
        hypparams["extra_knots"], (list, tuple)
    ):
        raise ValueError("Extra knots should be a list or tuple of values")

    _, B_age = create_splines(
        age_samples,
        df=hypparams["n_splines"],
        extra_knots=hypparams.get("extra_knots", None),
    )
    return {"bio": B_age}, age_samples, age_normalizer, age_unnormalizer


def initialize_params(
    counts,
    age,
    age_samples,
    age_normalizer,
    n_syllables,
    linear_model=False,
    n_splines=None,
    n_animals=None,
    n_sexes=None,
    n_animals_per_sex: Optional[list] = None,
    model_development=False,
    n_development_splines=None,
    n_sex_splines=None,
    zero_mean=True,
):
    if linear_model:
        weights = np.linalg.lstsq(
            np.stack([age_normalizer(age), np.ones_like(age)], axis=1),
            counts,
            rcond=None,
        )[0].T
        params = {"bio_basis_weights": jnp.array(weights)}
        return params

    # spline_class, _ = create_splines(age_samples, df=n_splines)
    # A = spline_class.transform(age_normalizer(age)).T

    # theta_list = []
    # for i in range(n_syllables):
    #     _theta = np.linalg.lstsq(A.T, (counts + 1)[:, i] * scale, rcond=-1)[0]
    #     theta_list.append(_theta)
    # theta_list = np.array(theta_list).T
    # theta_list = np.where(theta_list == 0, 1e-3, theta_list)
    # params = {"bio_basis_weights": jnp.log(jnp.array(theta_list.T))}
    params = {
        "bio_basis_weights": jnp.zeros((n_syllables, n_splines)),
        "size_slope": jnp.zeros((n_syllables, 1)),
    }

    # handles individuality params for models v3 and v4
    if n_animals_per_sex is not None:
        params["individual_biases"] = [
            jnp.zeros((n_syllables, x - 1)) for x in n_animals_per_sex
        ]
    elif n_animals is not None:
        params["individual_biases"] = jnp.zeros(
            (n_syllables, n_animals - 1 if zero_mean else n_animals)
        )

    if n_sexes is not None:
        params["sex_biases"] = jnp.zeros((n_syllables, n_sexes - 1))

    # handles model > v4 which adds a development/sex term
    if model_development and n_development_splines is None:
        # this is for the sigmoid
        params["development_weights"] = jnp.array((12, -5), dtype="float32").reshape(
            (1, 2)
        )
    elif n_development_splines is not None:
        params["development_weights"] = jnp.ones((1, n_development_splines))
    if n_sex_splines is not None:
        params["sex_weights"] = jnp.ones((1, n_sex_splines))

    return params


def compute_distribution_logprobs(
    counts, concentrations, true_age, sampled_age, hypparams
):
    counts, concentrations = mask_data(counts, concentrations, hypparams)

    total_counts = counts.sum(axis=1)

    dir_mult = tfp.distributions.DirichletMultinomial(
        total_count=total_counts[:, None],
        concentration=concentrations,
        name="DirichletMultinomial",
    )
    dir_probs = dir_mult.log_prob(counts[:, None, :])

    if hypparams.get("log_scale_sd", False):
        scale = scale_sd(hypparams["age_sd"], true_age) * hypparams["age_sd"]
        scale = scale[:, None]
    else:
        scale = hypparams["age_sd"]

    gauss_probs = tfp.distributions.Normal(
        loc=true_age[:, None], scale=scale, name="Normal"
    ).log_prob(sampled_age[None])

    return dir_probs + gauss_probs


def mask_data(counts, concentrations, hypparams):
    if "mask" in hypparams:
        concentrations = concentrations[
            hypparams["mask"][0][..., None],
            jnp.arange(concentrations.shape[1])[None, :, None],
            hypparams["mask"][1][:, None, :],
        ]
        counts = counts[hypparams["mask"]]
    return counts, concentrations


### ~~~~~~~~~~~ write a "linear" model in this framework ~~~~~~~~~~~ ###
def model_fun_v0(params, bases, age_samples, true_age, counts, hypparams):
    """
    parameter shapes:
        age_samples: shape (n_ages,)
        true_age: shape (n_sessions,)
        counts: shape (n_sessions, n_syllables)
    """
    pred_counts = params["bio_basis_weights"] @ jnp.stack(
        [age_samples, jnp.ones_like(age_samples)]
    )  # shape (n_syllables, n_ages)

    pred_counts = jnp.tile(pred_counts.T[None], (len(counts), 1, 1))
    # shape (n_sessions, n_ages, n_syllables)

    counts, pred_counts = mask_data(counts, pred_counts, hypparams)

    age_probs = tfp.distributions.Normal(loc=counts[:, None], scale=1).log_prob(
        pred_counts
    )

    gauss_probs = tfp.distributions.Normal(
        loc=true_age[:, None], scale=hypparams["age_sd"], name="Normal"
    ).log_prob(age_samples[None])

    return age_probs.sum(axis=-1) + gauss_probs


# ~~~~~~~~~~~ write the first spline regression model ~~~~~~~~~~~ #
def model_fun_v1(params, bases, age_samples, true_age, counts, hypparams):
    concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)

    concentrations = jnp.tile(concentrations.T[None], (len(counts), 1, 1))
    # shape (n_sessions, n_ages, n_syllables)

    # apply optional scaling parameter that alters the way the model is fit
    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


def swap_mask(hypparams):
    if "mask" in hypparams:
        hypparams["mask"], hypparams["heldout_mask"] = (
            hypparams["heldout_mask"],
            hypparams["mask"],
        )
    return hypparams


def neg_log_likelihood(
    params, bases, age_samples, true_age, counts, hypparams, model_fun, heldout=False
):
    # if heldout, use the heldout masks to compute likelihood
    if heldout:
        hypparams = swap_mask(hypparams)

    log_probs = model_fun(params, bases, age_samples, true_age, counts, hypparams)

    # if heldout, switch back mask values to original values
    if heldout:
        hypparams = swap_mask(hypparams)

    sample_lls = jax.scipy.special.logsumexp(log_probs, axis=1)

    # since zmax == 1 here, I just have to subtract the number of samples
    sample_lls = sample_lls - log_probs.shape[1]

    total_lls = sample_lls.sum()
    if heldout:
        return -total_lls

    params_ll = 0

    if "bio_params_sd" in hypparams and hypparams["bio_params_sd"] > 0:
        params_probs = tfp.distributions.Normal(
            loc=0, scale=hypparams["bio_params_sd"]
        ).log_prob(params["bio_basis_weights"])

        params_ll = params_probs.sum()

    total_lls = total_lls + params_ll
    return -total_lls


def create_splines(
    x, df=6, degree=3, include_intercept=True, extra_knots: Optional[list] = None
):
    bs = UnivariateBSplines(
        x, df=df, degree=degree, include_intercept=include_intercept, knots=extra_knots
    )
    return bs, bs.basis.T


def age_normalizer_factory(min_age=0, max_age=150, log_transform=False):
    if log_transform:
        if min_age <= 0:
            raise ValueError("Min age should be > 0")
        min_age = jnp.log(min_age)
        max_age = jnp.log(max_age)

    def age_normalizer(age):
        if log_transform:
            if jnp.sum(age <= 0) > 0:
                raise ValueError("Ages should be > 0")
            age = jnp.log(age)
        return (age - min_age) / (max_age - min_age)

    def age_unnormalizer(age):
        unnormalized_age = age * (max_age - min_age) + min_age
        if log_transform:
            unnormalized_age = jnp.exp(unnormalized_age)
        return unnormalized_age

    return age_normalizer, age_unnormalizer


def get_biological_age(
    params,
    bases,
    age_samples,
    chron_age,
    counts,
    hypparams,
    age_unnormalizer,
    model_fun,
):
    """Works with all models. Returns the predicted biological age."""
    hypparams = dissoc(hypparams, "mask")
    log_probs = model_fun(params, bases, age_samples, chron_age, counts, hypparams)
    predicted_bio_ages = age_samples[jnp.argmax(log_probs, axis=1)]
    return age_unnormalizer(predicted_bio_ages), jax.nn.softmax(log_probs, axis=1)


### ~~~~~~~~~~~ version 2 of the model - add size basis ~~~~~~~~~~~ ###


def model_fun_v2(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        params["size_slope"]
        @ bases["size"]  # make sure bases["size"] is shape (1, n_sessions)
    )  # shape (n_syllables, n_sessions)

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)

    concentrations = bio_concentrations + size_concentrations

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


### ~~~~~~~~~~~ version 3 of the model - add size basis and individuality terms ~~~~~~~~~~~ ###


def model_fun_v3(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        params["size_slope"]
        @ bases["size"]  # make sure bases["size"] is shape (1, n_sessions)
    )  # shape (n_syllables, n_sessions)

    individual_biases = (
        raise_dim(params["individual_biases"], axis=1) @ bases["individual"]
    )  # shape (n_syllables, n_sessions)

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)
    individual_biases = individual_biases.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)

    concentrations = bio_concentrations + size_concentrations + individual_biases

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


### ~~~~~~~~~~~ version 4 of the model - add size basis and individuality terms where individuality scales with age ~~~~~~~~~~~ ###


def model_fun_v4(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        params["size_slope"] @ bases["size"]
    )  # shape (n_syllables, n_sessions)

    bias_scale = jax.nn.sigmoid(
        params["development_weights"]
        @ jnp.stack([age_samples, jnp.ones_like(age_samples)], axis=0)
    )  # shape (1, n_ages)

    individual_biases = (
        raise_dim(params["individual_biases"], axis=1) @ bases["individual"]
    )  # shape (n_syllables, n_sessions)

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)
    individual_biases = individual_biases[
        :, None, :
    ].T  # shape (n_sessions, n_ages, n_syllables)

    concentrations = (
        bio_concentrations
        + size_concentrations
        + individual_biases * bias_scale[..., None]
    )

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


### ~~~~~~~~~~~ version 5 of the model - add size basis and individuality terms where individuality scales are learned ~~~~~~~~~~~ ###


def model_fun_v5(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        params["size_slope"] @ bases["size"]
    )  # shape (n_syllables, n_sessions)

    bias_scale = jnp.exp(
        params["development_weights"] @ bases["development"]
    )  # shape (1, n_ages)

    if hypparams.get("zero_mean", True):
        individual_biases = (
            raise_dim(params["individual_biases"], axis=1) @ bases["individual"]
        )  # shape (n_syllables, n_sessions)
    else:
        individual_biases = params["individual_biases"] @ bases["individual"]

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)
    individual_biases = individual_biases[
        :, None, :
    ].T  # shape (n_sessions, n_ages, n_syllables)

    concentrations = (
        bio_concentrations
        + size_concentrations
        + (individual_biases * bias_scale[..., None])
    )

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


### ~~~~~~~~~~~ version 6 of the model - add size basis, individuality, and sex terms where individuality, sex scales are learned ~~~~~~~~~~~ ###


def model_fun_v6(params, bases, age_samples, true_age, counts, hypparams):
    bio_concentrations = (
        params["bio_basis_weights"] @ bases["bio"]
    )  # shape (n_syllables, n_ages)
    size_concentrations = (
        params["size_slope"] @ bases["size"]
    )  # shape (n_syllables, n_sessions)

    ind_bias_scale = (
        params["development_weights"] @ bases["development"]
    )  # shape (1, n_ages)
    sex_bias_scale = params["sex_weights"] @ bases["sex"]  # shape (1, n_ages)

    # ind_bias_scale = jnp.exp(ind_bias_scale - ind_bias_scale.max())
    ind_bias_scale = jnp.exp(ind_bias_scale)
    # sex_bias_scale = jnp.exp(sex_bias_scale - sex_bias_scale.max())
    sex_bias_scale = jnp.exp(sex_bias_scale)

    # zero-mean within each sex
    individual_biases = (
        jnp.concatenate(
            [raise_dim(p, axis=1) for p in params["individual_biases"]], axis=1
        )
        @ bases["individual"]
    )  # shape (n_syllables, n_sessions)

    sex_biases = (
        raise_dim(params["sex_biases"], axis=1) @ bases["sex_id"]
    )  # shape (n_syllables, n_sessions)

    bio_concentrations = bio_concentrations.T[None]  # shape (1, n_ages, n_syllables)
    size_concentrations = size_concentrations.T[
        :, None, :
    ]  # shape (n_sessions, 1, n_syllables)
    individual_biases = (
        ind_bias_scale[..., None] * individual_biases[:, None, :].T
    )  # shape (n_sessions, n_ages, n_syllables)
    sex_biases = (
        sex_bias_scale[..., None] * sex_biases[:, None, :].T
    )  # shape (n_sessions, n_ages, n_syllables)

    concentrations = (
        bio_concentrations + size_concentrations + individual_biases + sex_biases
    )

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    log_probs = compute_distribution_logprobs(
        counts, concentrations, true_age, age_samples, hypparams
    )

    return log_probs


### ~~~~~~~~~~~ cross-validation section ~~~~~~~~~~~ ###
# add masking-based cross-validation
def masked_xval(
    features: dict,
    hypparams: dict,
    model_version: int,
    n_repeats: int,
    seed: int = 3,
    disable_tqdm: bool = False,
    return_loss_on_error=False,
):

    rng = np.random.RandomState(seed)
    init_components = model_setup(features, hypparams, model_version)

    output = {}
    for i in tqdm(range(n_repeats), disable=disable_tqdm):
        hypparams["mask"], hypparams["heldout_mask"] = create_masks(
            hypparams["n_keep_sylls"],
            hypparams["n_syllables"],
            hypparams["n_sessions"],
            rng,
        )
        output[i] = fit_model(
            features,
            hypparams,
            model_version,
            init_components,
            return_loss_on_error=return_loss_on_error,
        )
    return output


# add standard stratificed k-fold cross-validation
def stratified_xval(
    features: dict,
    hypparams: dict,
    model_version: int,
    n_folds: int,
    seed: int = 3,
    n_repeats: Optional[int] = None,
):

    hypparams = deepcopy(hypparams)

    init_components = model_setup(features, hypparams, model_version)

    if n_repeats is None:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        folds = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=seed
        )
    age_bins = pd.cut(features["ages"], bins=10, labels=False)

    output = {}
    for i, (train, test) in enumerate(folds.split(features["ages"], age_bins)):
        train_features = {
            "counts": features["counts"][train],
            "ages": features["ages"][train],
        }
        test_features = {
            "counts": features["counts"][test],
            "ages": features["ages"][test],
        }
        train_components = deepcopy(init_components)
        test_components = deepcopy(init_components)

        if model_version > 1:
            train_components.bases["size"] = init_components.bases["size"][:, train]
            test_components.bases["size"] = init_components.bases["size"][:, test]

            train_features["sizes"] = features["sizes"][train]
            test_features["sizes"] = features["sizes"][test]
        if model_version > 2:
            train_components.bases["individual"] = init_components.bases["individual"][
                :, train
            ]
            test_components.bases["individual"] = init_components.bases["individual"][
                :, test
            ]

            train_features["mice"] = features["mice"][train]
            test_features["mice"] = features["mice"][test]
        # if model_version == 5:
        #     train_components.bases["development"] = init_components.bases["development"]
        #     test_components.bases["development"] = init_components.bases["development"]

        train_components.true_age = init_components.true_age[train]
        test_components.true_age = init_components.true_age[test]

        out = fit_model(train_features, hypparams, model_version, train_components)
        heldout_nll = neg_log_likelihood(
            out["params"],
            test_components.bases,
            test_components.age_samples,
            test_components.true_age,
            test_features["counts"],
            hypparams,
            globals()[f"model_fun_v{model_version}"],
            heldout=True,
        )

        bio_ages, age_likelihoods = get_biological_age(
            out["params"],
            test_components.bases,
            test_components.age_samples,
            test_components.true_age,
            test_features["counts"],
            hypparams,
            test_components.age_unnormalizer,
            globals()[f"model_fun_v{model_version}"],
        )

        out_components = InitComponents(
            bases=valmap(np.array, test_components.bases),
            age_samples=np.array(test_components.age_samples),
            true_age=np.array(test_components.true_age),
            age_normalizer=test_components.age_normalizer,
            age_unnormalizer=test_components.age_unnormalizer,
            init=test_components.init,
            params=valmap(np.array, train_components.params),
            model_version=model_version,
        )

        out["heldout"] = {
            "heldout_ll": -int(heldout_nll),
            "bio_ages": bio_ages,
            "age_likelihoods": age_likelihoods,
            "test_components": out_components,
        }

        output[i] = out

    return output


@partial(jax.vmap, in_axes=(None, 1))
def concentration_interpolation(x, conc):
    return jnp.interp(x, jnp.linspace(0, 1, conc.shape[0]), conc)


@partial(jax.vmap, in_axes=(0, 0))
def expected_counts(concentrations, total_counts):
    return concentrations / concentrations.sum() * total_counts


def compute_heldout_r2(counts, predicted_counts, heldout_mask):
    total_mask = np.zeros(counts.shape, dtype=bool)
    total_mask[heldout_mask] = 1

    reorganized_counts = np.zeros_like(counts)
    reorganized_counts[heldout_mask] = predicted_counts[heldout_mask]

    keep_sylls = total_mask.sum(axis=0) > 3

    r2s = []
    _vars = []
    for syllable in np.where(keep_sylls)[0]:
        _r2 = r2_score(
            counts[total_mask[:, syllable], syllable],
            reorganized_counts[total_mask[:, syllable], syllable],
        )
        r2s.append(_r2)
        _var = (
            np.var(counts[total_mask[:, syllable], syllable])
            * total_mask[:, syllable].sum()
        )
        _vars.append(_var)

    weighted_r2 = np.average(r2s, weights=_vars)

    return np.array(r2s), weighted_r2


# straightforward model fitting function
def fit_model(
    features: dict,
    hypparams: dict,
    model_version: int,
    init_components: Optional[InitComponents] = None,
    return_loss_on_error=False,
):

    hypparams = deepcopy(hypparams)

    if init_components is None:
        init_components = model_setup(features, hypparams, model_version)

    if not hypparams.get("log_age", False):
        hypparams["age_sd"] = init_components.age_normalizer(
            hypparams["age_sd"] + hypparams["min_age"]
        )

    ll_fun = partial(
        neg_log_likelihood,
        bases=init_components.bases,
        age_samples=init_components.age_samples,
        true_age=init_components.true_age,
        counts=features["counts"],
        hypparams=hypparams,
        model_fun=globals()[f"model_fun_v{model_version}"],
        heldout=False,
    )
    optimized_params, loss = optimize(
        init_components.params,
        ll_fun,
        lr=hypparams["lr"],
        n_iter=hypparams["n_opt_iter"],
    )

    heldout_nll = neg_log_likelihood(
        optimized_params,
        init_components.bases,
        init_components.age_samples,
        init_components.true_age,
        features["counts"],
        hypparams=hypparams,
        model_fun=globals()[f"model_fun_v{model_version}"],
        heldout=True,
    )

    bio_ages, age_likelihoods = get_biological_age(
        optimized_params,
        init_components.bases,
        init_components.age_samples,
        init_components.true_age,
        features["counts"],
        hypparams,
        init_components.age_unnormalizer,
        globals()[f"model_fun_v{model_version}"],
    )

    concentrations, concentration_components = compute_concentrations(
        optimized_params, model_version, init_components, hypparams
    )

    norm_bio_age = init_components.age_normalizer(bio_ages)

    concs = []
    for i in range(len(bio_ages)):
        idx = i if len(concentrations) > 1 else 0
        _concs = concentration_interpolation(norm_bio_age[idx], concentrations[idx])
        concs.append(_concs)
    concs = jnp.array(concs)

    extra_output = {}
    if model_version > 0:
        total_counts = features["counts"].sum(axis=1)
        predicted_counts = expected_counts(concs, total_counts)

        if "heldout_mask" in hypparams:
            heldout_counts = features["counts"][hypparams["heldout_mask"]]

            try:
                extra_output = {
                    "heldout_r2_total_v2": r2_score(
                        heldout_counts,
                        predicted_counts[hypparams["heldout_mask"]],
                        multioutput="variance_weighted",
                    ),
                    "heldout_r2_total_v3": compute_heldout_r2(
                        features["counts"], predicted_counts, hypparams["heldout_mask"]
                    )[1],
                }
            except ValueError:
                print(np.isnan(predicted_counts).sum())

    else:
        predicted_counts = concs
        if "heldout_mask" in hypparams:
            heldout_counts = features["counts"][hypparams["heldout_mask"]]

            extra_output = {
                "heldout_r2_total": r2_score(
                    heldout_counts,
                    predicted_counts[hypparams["heldout_mask"]],
                    multioutput="variance_weighted",
                ),
                "heldout_r2_total_v3": compute_heldout_r2(
                    features["counts"], predicted_counts, hypparams["heldout_mask"]
                )[1],
            }

    out_components = InitComponents(
        bases=valmap(np.array, init_components.bases),
        age_samples=np.array(init_components.age_samples),
        true_age=np.array(init_components.true_age),
        age_normalizer=init_components.age_normalizer,
        age_unnormalizer=init_components.age_unnormalizer,
        init=init_components.init,
        params=valmap(np.array, optimized_params),
        model_version=model_version,
    )

    try:
        return {
            "params": valmap(np.array, optimized_params),
            "heldout_ll": -int(heldout_nll),
            "loss": np.array(loss),
            "bio_ages": np.array(bio_ages),
            "age_likelihoods": np.array(age_likelihoods),
            "true_ages": np.array(
                init_components.age_unnormalizer(init_components.true_age)
            ),
            "init_components": out_components,
            "concentrations": np.array(concentrations),
            "concentration_components": valmap(np.array, concentration_components),
            "predicted_counts": np.array(predicted_counts),
            "counts": np.array(features["counts"]),
            "r2_total": r2_score(
                np.array(features["counts"]),
                np.array(predicted_counts),
                multioutput="variance_weighted",
            ),
            "r2_each": r2_score(
                np.array(features["counts"]),
                np.array(predicted_counts),
                multioutput="raw_values",
            ),
            **extra_output,
        }
    except ValueError as e:
        if return_loss_on_error:
            return {"loss": np.array(loss)}
        else:
            raise


def compute_concentration_components(
    params, model_version, init_components: InitComponents
) -> dict:
    concentrations = dict()
    if model_version == 0:
        bio_concentrations = params["bio_basis_weights"] @ jnp.stack(
            [init_components.age_samples, jnp.ones_like(init_components.age_samples)]
        )
        concentrations["bio"] = bio_concentrations
    if model_version > 0:
        bio_concentrations = (
            params["bio_basis_weights"] @ init_components.bases["bio"]
        )  # shape (n_syllables, n_ages)
        concentrations["bio"] = bio_concentrations
    if model_version > 1:
        size_concentrations = (
            params["size_slope"] @ init_components.bases["size"]
        )  # shape (n_syllables, n_sessions)
        concentrations["size"] = size_concentrations
    if model_version > 2 and model_version < 6:
        try:
            individual_biases = (
                raise_dim(params["individual_biases"], axis=1)
                @ init_components.bases["individual"]
            )  # shape (n_syllables, n_sessions)
        except TypeError:
            individual_biases = (
                params["individual_biases"] @ init_components.bases["individual"]
            )

        concentrations["indiv"] = (
            individual_biases  # multiplied with total concentrations
        )
    if model_version == 4:
        bias_scale = jax.nn.sigmoid(
            params["development_weights"]
            @ jnp.stack(
                [
                    init_components.age_samples,
                    jnp.ones_like(init_components.age_samples),
                ],
                axis=0,
            )
        )  # shape (1, n_ages)
        concentrations["indiv"] = concentrations["indiv"][:, None, :]
        concentrations["indiv_scale"] = bias_scale.squeeze()
    elif model_version == 5:
        bias_scale = jnp.exp(
            params["development_weights"] @ init_components.bases["development"]
        )
        concentrations["indiv"] = concentrations["indiv"][:, None, :]
        concentrations["indiv_scale"] = bias_scale.squeeze()
    elif model_version == 6:
        individual_biases = (
            jnp.concatenate(
                [raise_dim(p, axis=1) for p in params["individual_biases"]], axis=1
            )
            @ init_components.bases["individual"]
        )
        concentrations["indiv"] = individual_biases[
            :, None, :
        ]  # multiplied with total concentrations

        indiv_scale = (
            params["development_weights"] @ init_components.bases["development"]
        )
        indiv_scale = jnp.exp(indiv_scale)
        concentrations["indiv_scale"] = indiv_scale.squeeze()

        sex_biases = (
            raise_dim(params["sex_biases"], axis=1) @ init_components.bases["sex_id"]
        )
        concentrations["sex"] = sex_biases[:, None, :]

        sex_scale = params["sex_weights"] @ init_components.bases["sex"]
        sex_scale = jnp.exp(sex_scale)

        concentrations["sex_scale"] = sex_scale.squeeze()

    return concentrations


def compute_concentrations(
    params, model_version, init_components: InitComponents, hypparams
) -> Tuple[jnp.array, dict]:
    concentration_components = compute_concentration_components(
        params, model_version, init_components
    )

    concentrations = concentration_components["bio"].T[None]
    if model_version > 1:
        concentrations = concentrations + concentration_components["size"].T[:, None, :]
    if model_version == 3:
        concentrations = (
            concentrations + concentration_components["indiv"].T[:, None, :]
        )
    elif model_version > 3:
        concentrations = (
            concentrations
            + concentration_components["indiv"].T
            * concentration_components["indiv_scale"][None, :, None]
        )
    if model_version == 6:
        concentrations = (
            concentrations
            + concentration_components["sex"].T
            * concentration_components["sex_scale"][None, :, None]
        )

    if len(concentrations) == 1:
        concentrations = jnp.tile(
            concentrations, (init_components.true_age.shape[0], 1, 1)
        )

    if model_version == 0:
        return concentrations, concentration_components

    if "scale" in hypparams and hypparams.get("normalize_by_softmax", False):
        concentrations = jax.nn.softmax(concentrations, axis=2) * hypparams["scale"]
    else:
        concentrations = jnp.exp(concentrations)

    return concentrations, concentration_components


# ~~~~~~~~ add ability to perform constrained optimization ~~~~~~~~ #


# constrained optimization: fit a subset of parameters on one dataset and fix them -
# then, fit the rest of the parameters on a different dataset
def constrained_model(params, constrained_params, ll_fun):
    return ll_fun(merge(params, constrained_params))


def constrained_fit(
    features: dict,
    constrained_params: dict,
    hypparams: dict,
    model_version: int,
    return_loss_on_error=False,
    init_components=None,
):
    hypparams = deepcopy(hypparams)

    if init_components is None:
        init_components = model_setup(features, hypparams, model_version)
    else:
        init_components_copy = deepcopy(init_components)
        init_components = model_setup(features, hypparams, model_version)
        init_components.age_normalizer = init_components_copy.age_normalizer
        init_components.age_unnormalizer = init_components_copy.age_unnormalizer

    if not hypparams.get("log_age", False):
        hypparams["age_sd"] = init_components.age_normalizer(
            hypparams["age_sd"] + hypparams["min_age"]
        )

    model_fun = globals()[f"model_fun_v{model_version}"]

    ll_fun = partial(
        neg_log_likelihood,
        bases=init_components.bases,
        age_samples=init_components.age_samples,
        true_age=init_components.true_age,
        counts=features["counts"],
        hypparams=hypparams,
        model_fun=model_fun,
        heldout=False,
    )

    ll_fun = partial(
        constrained_model, constrained_params=constrained_params, ll_fun=ll_fun
    )

    init_components.params = blacklist(init_components.params, list(constrained_params))

    optimized_params, loss = optimize(
        init_components.params,
        ll_fun,
        lr=hypparams["lr"],
        n_iter=hypparams["n_opt_iter"],
    )

    # add in the constrained parameters for later functions
    optimized_params = merge(optimized_params, constrained_params)

    heldout_nll = neg_log_likelihood(
        optimized_params,
        init_components.bases,
        init_components.age_samples,
        init_components.true_age,
        features["counts"],
        hypparams=hypparams,
        model_fun=model_fun,
        heldout=True,
    )

    bio_ages, age_likelihoods = get_biological_age(
        optimized_params,
        init_components.bases,
        init_components.age_samples,
        init_components.true_age,
        features["counts"],
        hypparams,
        init_components.age_unnormalizer,
        model_fun,
    )

    concentrations, concentration_components = compute_concentrations(
        optimized_params, model_version, init_components, hypparams
    )

    norm_bio_age = init_components.age_normalizer(bio_ages)

    concs = []
    for i in range(len(bio_ages)):
        idx = i if len(concentrations) > 1 else 0
        _concs = concentration_interpolation(norm_bio_age[idx], concentrations[idx])
        concs.append(_concs)
    concs = jnp.array(concs)

    extra_output = {}
    if model_version > 0:
        total_counts = features["counts"].sum(axis=1)
        predicted_counts = expected_counts(concs, total_counts)

        if "heldout_mask" in hypparams:
            heldout_counts = features["counts"][hypparams["heldout_mask"]]

            try:
                extra_output = {
                    "heldout_r2_total_v2": r2_score(
                        heldout_counts,
                        predicted_counts[hypparams["heldout_mask"]],
                        multioutput="variance_weighted",
                    ),
                    "heldout_r2_total_v3": compute_heldout_r2(
                        features["counts"], predicted_counts, hypparams["heldout_mask"]
                    )[1],
                }
            except ValueError:
                print(np.isnan(predicted_counts).sum())

    else:
        predicted_counts = concs
        if "heldout_mask" in hypparams:
            heldout_counts = features["counts"][hypparams["heldout_mask"]]

            extra_output = {
                "heldout_r2_total": r2_score(
                    heldout_counts,
                    predicted_counts[hypparams["heldout_mask"]],
                    multioutput="variance_weighted",
                ),
                "heldout_r2_total_v3": compute_heldout_r2(
                    features["counts"], predicted_counts, hypparams["heldout_mask"]
                )[1],
            }

    out_components = InitComponents(
        bases=valmap(np.array, init_components.bases),
        age_samples=np.array(init_components.age_samples),
        true_age=np.array(init_components.true_age),
        age_normalizer=init_components.age_normalizer,
        age_unnormalizer=init_components.age_unnormalizer,
        init=init_components.init,
        params=valmap(np.array, optimized_params),
        model_version=model_version,
    )

    try:
        return {
            "params": valmap(np.array, optimized_params),
            "heldout_ll": -int(heldout_nll),
            "loss": np.array(loss),
            "bio_ages": np.array(bio_ages),
            "age_likelihoods": np.array(age_likelihoods),
            "true_ages": np.array(
                init_components.age_unnormalizer(init_components.true_age)
            ),
            "init_components": out_components,
            "concentrations": np.array(concentrations),
            "concentration_components": valmap(np.array, concentration_components),
            "predicted_counts": np.array(predicted_counts),
            "counts": np.array(features["counts"]),
            "r2_total": r2_score(
                features["counts"], predicted_counts, multioutput="variance_weighted"
            ),
            "r2_each": r2_score(
                features["counts"], predicted_counts, multioutput="raw_values"
            ),
            **extra_output,
        }
    except ValueError as e:
        if return_loss_on_error:
            return {"loss": np.array(loss)}
        else:
            raise
