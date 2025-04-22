import torch
import numpy as np


def sampling(mean, logvar, n_samples=1):
    mu = torch.stack([mean] * n_samples, dim=-1)
    sigma = torch.stack([torch.exp(logvar * 0.5)] * n_samples, dim=-1)
    eps = torch.zeros_like(sigma).normal_()
    zs = eps * sigma + mu
    return zs


def log_pdf(mean, logvar, zs) -> torch.Tensor:
    if len(zs.shape) == len(mean.shape) + 1:
        mean = mean.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)
    return -0.5 * np.log(2 * np.pi) - 0.5 * logvar - \
           (zs - mean).pow(2) / (2 * torch.exp(logvar) + 1e-4)


def exp_mean_log(tensor, dim, weights=None):
    max_values = tensor.max(dim=dim, keepdims=True).values
    tensor = tensor - max_values
    if weights is None:
        tensor = tensor.exp().sum(dim=dim, keepdims=True).log() - np.log(tensor.shape[dim])
    else:
        tensor = (tensor.exp() * weights).sum(dim=dim, keepdims=True).log()
    tensor = tensor + max_values
    return tensor.squeeze(dim=dim)


anneal_steps = 0
def get_anneal_kl_weight(p1=0.0, p2=0.5, train_steps=1000, eval_steps=0):
    # [0, p1]: beta = 0
    # [p1, p2]: beta from 0 to 1
    # [p2, 1]: beta = 1
    global anneal_steps
    anneal_steps += 1
    anneal_cycle = train_steps + eval_steps
    anneal_phase = anneal_steps % anneal_cycle
    if anneal_phase <= eval_steps:
        beta = 1.0
    else:
        p = (anneal_phase - eval_steps) / train_steps
        beta = min(max((p-p1) / (p2-p1), 0), 1)
    return beta

def compute_kl_penalty(mean, logvar, vae_type="vanilla", **kwargs):
    batch_size, dim_z = mean.shape
    if vae_type == "Vanilla" or vae_type == "EMB-VAE":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
    elif vae_type == "Anneal":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        kl_penalty = kl_penalty * get_anneal_kl_weight()
    elif vae_type == "Anneal2":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        kl_penalty = kl_penalty * get_anneal_kl_weight(p1=0.5, p2=0.75)
    elif vae_type == "AE":
        kl_penalty = mean.sum().detach().fill_(0)
    elif vae_type == "Beta-VAE" or vae_type == "Beta01-VAE":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        kl_penalty = kl_penalty * kwargs.get("beta", 0.1)
    elif vae_type == "Beta02-VAE":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        kl_penalty = kl_penalty * kwargs.get("beta", 0.2)
    elif vae_type == "Beta05-VAE":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1).mean(dim=0)
        kl_penalty = kl_penalty * kwargs.get("beta", 0.5)
    elif vae_type == "FB-VAE":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_penalty = torch.max(kl_penalty, kl_penalty.clone().fill_(kwargs.get("kl_norm", 0.5)))
        kl_penalty = kl_penalty.sum(dim=1).mean(dim=0)
    elif vae_type == "Optimus":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_penalty = torch.max(kl_penalty, kl_penalty.clone().fill_(kwargs.get("kl_norm", 0.5)))
        kl_penalty = kl_penalty * get_anneal_kl_weight(p1=0.5, p2=0.75)
        kl_penalty = kl_penalty.sum(dim=1).mean(dim=0)
    elif vae_type == "Optimus_":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_penalty = torch.max(kl_penalty, kl_penalty.clone().fill_(kwargs.get("kl_norm", 0.5)))
        kl_penalty = kl_penalty * get_anneal_kl_weight(p1=0.5, p2=0.75, train_steps=kwargs["train_steps"], eval_steps=kwargs["eval_steps"])
        kl_penalty = kl_penalty.sum(dim=1).mean(dim=0)
    elif vae_type == "Anneal_":
        kl_penalty = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)
        kl_penalty = kl_penalty * get_anneal_kl_weight(p1=0.5, p2=0.75, train_steps=kwargs["train_steps"], eval_steps=kwargs["eval_steps"])
        kl_penalty = kl_penalty.sum(dim=1).mean(dim=0)
    elif vae_type == "DG-VAE" or vae_type == "EMB-DG-VAE":
        n_samples = kwargs.get("kl_sampling_times", 16)
        post_samples = sampling(mean, logvar, n_samples).transpose(1,2) # [batch_size, n_samples, dim_z]
        post_agg_samples = post_samples.reshape(batch_size*n_samples, dim_z) # share inside batch
        post_agg_log_probs = log_pdf(
            mean=mean[None,:,:],
            logvar=logvar[None,:,:],
            zs=post_agg_samples[:,None,:]
        ) # [batch_size*n_samples, agg_size, dim_z]
        post_agg_log_probs = exp_mean_log(post_agg_log_probs, dim=1) # aggregation
        prior_log_probs = log_pdf(
            mean=torch.zeros_like(mean)[None,:,:],
            logvar=torch.zeros_like(logvar)[None,:,:],
            zs=post_agg_samples[:,None,:]
        ) # [batch_size*n_samples, agg_size, dim_z]
        prior_log_probs = exp_mean_log(prior_log_probs, dim=1) # aggregation
        kl_penalty = (post_agg_log_probs - prior_log_probs).sum(dim=1).mean(dim=0)
    else:
        raise NotImplementedError(vae_type)
    return kl_penalty


def compute_mi(mean, logvar, n_samples=16):
    N = mean.shape[0]
    zs = sampling(mean, logvar, n_samples)
    log_q_yz_x = log_pdf(mean, logvar, zs).sum(dim=1) - np.log(N)
    log_q_y_x = - np.log(N)
    log_q_z_x = exp_mean_log(
        log_pdf(mean[None, :, :], logvar[None, :, :], zs[:, None, :, :]).sum(dim=2), dim=1)
    mi = (log_q_yz_x - log_q_y_x - log_q_z_x).mean(dim=0).mean(dim=-1)
    return mi