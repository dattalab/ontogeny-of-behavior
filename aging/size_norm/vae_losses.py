import torch
import numpy as np

LN2PI = np.log(2 * np.pi)


# all losses copied from behavenet:
# https://behavenet.readthedocs.io/en/develop/api/behavenet.models.vaes.PSVAE.html
def gaussian_ll(y_pred, y_mean, masks=None, std=1):
    """Compute multivariate Gaussian log-likelihood with a fixed diagonal noise covariance matrix.
    Parameters
    ----------
    y_pred : :obj:`torch.Tensor`
        predicted data of shape (n_frames, ...)
    y_mean : :obj:`torch.Tensor`
        true data of shape (n_frames, ...)
    masks : :obj:`torch.Tensor`, optional
        binary mask that is the same size as `y_pred` and `y_true`; by placing 0 entries in the
        mask, the corresponding dimensions will not contribute to the loss term, and will therefore
        not contribute to parameter updates
    std : :obj:`float`, optional
        fixed standard deviation for all dimensions in the multivariate Gaussian
    Returns
    -------
    :obj:`torch.Tensor`
        Gaussian log-likelihood summed across dims, averaged across batch
    """
    dims = y_pred.shape
    n_dims = np.prod(dims[1:])  # first value is n_frames in batch
    log_var = np.log(std ** 2)

    if masks is not None:
        diff_sq = ((y_pred - y_mean) ** 2) * masks
    else:
        diff_sq = (y_pred - y_mean) ** 2

    ll = -(0.5 * LN2PI + 0.5 * log_var) * n_dims - (0.5 / (std ** 2)) * diff_sq.sum(
        axis=tuple(1 + np.arange(len(dims[1:])))
    )

    return torch.mean(ll)


def decomposed_kl(z, mu, logvar):
    """Decompose KL term in VAE loss.
    Decomposes the KL divergence loss term of the variational autoencoder into three terms:
    1. index code mutual information
    2. total correlation
    3. dimension-wise KL
    None of these terms can be computed exactly when using stochastic gradient descent. This
    function instead computes approximations as detailed in https://arxiv.org/pdf/1802.04942.pdf.
    Parameters
    ----------
    z : :obj:`torch.Tensor`
        sample of shape (n_frames, n_dims)
    mu : :obj:`torch.Tensor`
        mean parameter of shape (n_frames, n_dims)
    logvar : :obj:`torch.Tensor`
        log variance parameter of shape (n_frames, n_dims)
    Returns
    -------
    :obj:`tuple`
        - index code mutual information (:obj:`torch.Tensor`)
        - total correlation (:obj:`torch.Tensor`)
        - dimension-wise KL (:obj:`torch.Tensor`)
    """

    # Compute log(q(z(x_j)|x_i)) for every sample/dimension in the batch, which is a tensor of
    # shape (n_frames, n_dims). In the following comments, (n_frames, n_frames, n_dims) are indexed
    # by [j, i, l].
    #
    # Note that the insertion of `None` expands dims to use torch's broadcasting feature
    # z[:, None]: (n_frames, 1, n_dims)
    # mu[None, :]: (1, n_frames, n_dims)
    # logvar[None, :]: (1, n_frames, n_dims)
    log_qz_prob = _gaussian_log_density_unsummed(z[:, None], mu[None, :], logvar[None, :])

    # Compute log(q(z(x_j))) as
    # log(sum_i(q(z(x_j)|x_i))) + constant
    # = log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant
    # = log(sum_i(exp(sum_l log q(z(x_j)_l|x_i))) + constant (assumes q is factorized)
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2, keepdim=False),  # sum over gaussian dims
        dim=1,  # logsumexp over batch
        keepdim=False,
    )

    # Compute log prod_l q(z(x_j)_l | x_j)
    # = sum_l log q(z(x_j)_l | x_j)
    log_qz_ = torch.diag(torch.sum(log_qz_prob, dim=2, keepdim=False))  # sum over gaussian dims

    # Compute log prod_l p(z(x_j)_l)
    # = sum_l(log(sum_i(q(z(x_j)_l|x_i))) + constant
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False),  # logsumexp over batch
        dim=1,  # sum over gaussian dims
        keepdim=False,
    )

    # Compute sum_l log p(z(x_j)_l)
    log_pz_prob = _gaussian_log_density_unsummed_std_normal(z)
    log_pz_product = torch.sum(log_pz_prob, dim=1, keepdim=False)  # sum over gaussian dims

    idx_code_mi = torch.mean(log_qz_ - log_qz)
    total_corr = torch.mean(log_qz - log_qz_product)
    dim_wise_kl = torch.mean(log_qz_product - log_pz_product)

    return idx_code_mi, total_corr, dim_wise_kl


def _gaussian_log_density_unsummed(z, mu, logvar):
    """First step of Gaussian log-density computation, without summing over dimensions.
    Assumes a diagonal noise covariance matrix.
    """
    diff_sq = (z - mu) ** 2
    inv_var = torch.exp(-logvar)
    return -0.5 * (inv_var * diff_sq + logvar + LN2PI)


def _gaussian_log_density_unsummed_std_normal(z):
    """First step of Gaussian log-density computation, without summing over dimensions.
    Assumes a diagonal noise covariance matrix.
    """
    diff_sq = z ** 2
    return -0.5 * (diff_sq + LN2PI)


def vae_loss(
    y_hat, y_true, beta=5, kl_factor=1, gauss_std=0.075
) -> tuple[torch.Tensor, dict]:
    y_hat, sample, mu, logvar = y_hat
    index_code_mi, total_corr, dim_wise_kl = decomposed_kl(sample, mu, logvar)

    loss_dict = {
        "gauss": -gaussian_ll(y_hat, y_true, std=gauss_std),
        "index_code_mi": kl_factor * index_code_mi,
        "total_correlation": beta * total_corr,
        "dim_wise_kl": kl_factor * dim_wise_kl,
    }
    total_loss = sum(loss_dict.values())
    loss_dict["total_loss"] = total_loss
    loss_dict["mse_loss"] = torch.nn.functional.mse_loss(y_hat, y_true)
    return total_loss, loss_dict