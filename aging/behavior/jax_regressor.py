import jax
import optax
import jax.numpy as jnp
import pandas as pd
from optax._src import base
from toolz import partial
from inspect import signature
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_pred - y_true))


def mae(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))


def model_loss_l1(coef, x, y_true, model, l1=0.5):
    y_pred = model(x, **coef).squeeze()
    l1_coef = jnp.mean(jnp.abs(coef["submodel_weights"]))
    return (1 - l1) * mse(y_true, y_pred) + l1 * l1_coef


def model_loss(coef, x, y_true, model):
    y_pred = model(x, **coef).squeeze()
    return mse(y_true, y_pred)


def nonneg_params():

    def init_fn(params):
        return base.EmptyState()

    def update_fn(updates, state, params):
        if 'submodel_weights' in params:
            p = params['submodel_weights']
            u = updates['submodel_weights']
            updates['submodel_weights'] = jnp.where((p + u) < 0, 1e-8 - p, u)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def _optimize(coef, model, X, y, loss_fn, lr=0.001, n_iter=1000, jit=True):
    optimizer = optax.chain(
        optax.adam(learning_rate=lr),
        nonneg_params(),
    )
    opt_state = optimizer.init(coef)
    if jit:
        val_grad = jax.jit(jax.value_and_grad(partial(loss_fn, model=model)))
    else:
        val_grad = jax.value_and_grad(partial(loss_fn, model=model))

    def step(state, i):
        opt_state, params = state
        loss, grad = val_grad(params, X, y)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), (loss, grad)

    (opt_state, params), loss = jax.lax.scan(step, (opt_state, coef), jnp.arange(n_iter))

    return params, loss


class JaxRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        model,
        loss_fun,
        coef_shapes,
        jit=True,
        seed=0,
        l1=0.5,
        n_iter=100,
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.jit = jit
        self.seed = seed
        self.l1 = l1
        self.n_iter = n_iter
        self.model = model
        self.loss_fun = loss_fun
        self.lr = lr
        self.coef_shapes = coef_shapes
        self.initialize(coef_shapes)

    def initialize(self, coef_shape: dict):
        """Initialize coefficients for the model
        initialize a dictionary of coefficient vectors,
            where each value is a tuple of the shape of the coefficient vector
        """
        rng = jax.random.PRNGKey(self.seed)

        def recursive_init(x, rng):
            if isinstance(x, (tuple, list)):
                return jax.random.uniform(key=rng, maxval=0.2, shape=x)
            elif isinstance(x, dict):
                return {
                    k: recursive_init(v, rng)
                    for (k, v), rng in zip(x.items(), jax.random.split(rng, len(x)))
                }

        coef = recursive_init(coef_shape, rng)
        self.coef_ = coef

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X_ = jnp.array(X, dtype=jnp.float32)
        y_ = jnp.array(y, dtype=jnp.float32)

        if hasattr(self, "is_fitted_") and self.is_fitted_:
            self.initialize(self.coef_shapes)

        if "l1" in signature(self.loss_fun).parameters:
            loss_fun = partial(self.loss_fun, l1=self.l1)
        else:
            loss_fun = self.loss_fun

        self.coef_, (self.loss_, self.grads_) = _optimize(
            self.coef_,
            self.model,
            X_,
            y_,
            loss_fun,
            jit=self.jit,
            n_iter=self.n_iter,
            lr=self.lr,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model(X, **self.coef_)

    def gradient(self, X):
        def fn(x):
            return self.model(x, **self.coef_).squeeze()
        grad = jax.grad(fn)
        return jnp.array([grad(x) for x in X.squeeze()])


def exponential(X, a, b, c, offset):
    return a * jnp.exp(b * X + c) + offset


def linear(X, m, b):
    return m * X + b


def logistic(X, m, b, offset, scale):
    return scale / (1 + jnp.exp(-(m * X + b))) + offset


def quadratic(X, a, b, c):
    return a * jnp.square(X) + b * X + c


def multi_model_function(
    X,
    exponential_params,
    linear_params,
    quadratic_params,
    logistic_params,
    submodel_weights,
):
    """Apply parameters for combining multiple model families to a
    single syllable to predict age
    """
    exp_hat = exponential(X, **exponential_params)
    lin_hat = linear(X, **linear_params)
    logistic_hat = logistic(X, **logistic_params)
    quad_hat = quadratic(X, **quadratic_params)
    combined = jnp.concatenate([exp_hat, lin_hat, logistic_hat, quad_hat], axis=1)
    return jnp.dot(combined, submodel_weights)


if __name__ == "__main__":
    # test the model
    x = jnp.linspace(-2, 2, 400)
    a = 1
    b = 0.05
    c = 1.2
    offset = 0.5
    y = a * jnp.exp(b * x + c) + offset

    coef_shapes = dict(a=(1, 1), b=(1, 1), c=(1, 1), offset=(1, 1))
    mdl = JaxRegressor(
        model=exponential,
        loss_fun=model_loss,
        coef_shapes=coef_shapes,
        jit=True,
        seed=0,
        n_iter=10_000,
        lr=5e-3,
    )
    mdl.fit(x[:, None], y)
    print("True parameters:", f"a={a}, b={b}, c={c}, offset={offset}")
    print(
        "Estimated parameters:", {k: round(v.item(), 2) for k, v in mdl.coef_.items()}
    )
