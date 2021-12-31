import torch
import numpy as np
import torch.nn.functional as F

# pair x+sigma*z with x-sigma*x
# updated to improve numerical stability
def hosm_low_rank(score1, score2, samples, sigma=0.01):
    vectors = torch.randn_like(samples)  # z in the notes
    n, dim = vectors.shape

    perturbed_inputs = samples + vectors * sigma

    s2_1 = score2(perturbed_inputs).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_1 = score1(perturbed_inputs.reshape(-1, 1, 28, 28))
    s1_product_1 = torch.einsum('ij, ik -> ijk', s1_1, s1_1)
    h_1 = (s2_1 + s1_product_1).view(n, -1)

    vectors_product = torch.einsum('ij, ik -> ijk', vectors, vectors)
    eye = torch.eye(dim, device=vectors.device)
    eye = eye.unsqueeze(0)
    eye = eye[None, ...]
    diff = (eye - vectors_product) / (sigma ** 2)

    loss = h_1 ** 2 + 2 * diff.view(n, -1) * h_1
    loss = loss.sum(dim=-1)
    loss = loss.mean(dim=0) / 2.

    return loss

# pair x+sigma*z with x-sigma*x
# updated to improve numerical stability
def hosm_plus_vr_low_rank(score1, score2, samples, sigma=0.01):
    vectors = torch.randn_like(samples)  # z in the notes
    n, dim = vectors.shape

    perturbed_inputs1 = samples + vectors * sigma
    perturbed_inputs2 = samples - vectors * sigma

    # x+sigma*z
    s2_1 = score2(perturbed_inputs1).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_1 = score1(perturbed_inputs1)
    s1_product_1 = torch.einsum('ij, ik -> ijk', s1_1, s1_1)
    h_1 = (s2_1 + s1_product_1).view(n, -1)

    # x-sigma*z
    s2_2 = score2(perturbed_inputs2).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_2 = score1(perturbed_inputs2)
    s1_product_2 = torch.einsum('ij, ik -> ijk', s1_2, s1_2)
    h_2 = (s2_2 + s1_product_2).view(n, -1)

    # (I - z*z^T) / sigma ** 2
    vectors_product = torch.einsum('ij, ik -> ijk', vectors, vectors)
    eye = torch.eye(dim, device=vectors.device)
    eye = eye.unsqueeze(0)
    eye = eye.repeat(n, 1, 1)
    diff = (eye - vectors_product) / (sigma ** 2)

    s2_vr = score2(samples).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_vr = score1(samples)
    s1_product_vr = torch.einsum('ij, ik -> ijk', s1_vr, s1_vr)
    h_vr = (s2_vr + s1_product_vr).view(n, -1)

    loss = (h_1 ** 2 + h_2 ** 2) + 2 * diff.view(n, -1) * ((h_1 - h_vr) + (h_2 - h_vr))
    loss = loss.sum(dim=-1)
    loss = loss.mean(dim=0) / 2.

    return loss

# pair x+sigma*z with x-sigma*x
# updated to improve numerical stability
def hosm(score1, score2, samples, sigma=0.01):
    vectors = torch.randn_like(samples)  # z in the notes
    n, dim = vectors.shape
    perturbed_inputs = samples + vectors * sigma

    s2_1 = score2(perturbed_inputs).reshape(samples.shape[0], -1)
    s2_1 = torch.diag_embed(s2_1, offset=0, dim1=-2, dim2=-1).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_1 = score1(perturbed_inputs.reshape(-1, 1, 28, 28))
    s1_product_1 = torch.einsum('ij, ik -> ijk', s1_1, s1_1)
    h_1 = (s2_1 + s1_product_1).view(n, -1)

    vectors_product = torch.einsum('ij, ik -> ijk', vectors, vectors)
    eye = torch.eye(dim, device=vectors.device)
    eye = eye.unsqueeze(0)
    eye = eye[None, ...]
    diff = (eye - vectors_product) / (sigma ** 2)

    loss = h_1 ** 2 + 2 * diff.view(n, -1) * h_1
    loss = loss.sum(dim=-1)
    loss = loss.mean(dim=0) / 2.

    return loss

# pair x+sigma*z with x-sigma*x
# updated to improve numerical stability
def hosm_plus_vr(score1, score2, samples, sigma=0.01):
    vectors = torch.randn_like(samples)  # z in the notes
    n, dim = vectors.shape

    perturbed_inputs1 = samples + vectors * sigma
    perturbed_inputs2 = samples - vectors * sigma

    # x+sigma*z
    s2_1 = score2(perturbed_inputs1).reshape(samples.shape[0], -1)
    s2_1 = torch.diag_embed(s2_1, offset=0, dim1=-2, dim2=-1).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_1 = score1(perturbed_inputs1)
    s1_product_1 = torch.einsum('ij, ik -> ijk', s1_1, s1_1)
    h_1 = (s2_1 + s1_product_1).view(n, -1)

    # x-sigma*z
    s2_2 = score2(perturbed_inputs2).reshape(samples.shape[0], -1)
    s2_2 = torch.diag_embed(s2_2, offset=0, dim1=-2, dim2=-1).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_2 = score1(perturbed_inputs2)
    s1_product_2 = torch.einsum('ij, ik -> ijk', s1_2, s1_2)
    h_2 = (s2_2 + s1_product_2).view(n, -1)

    # (I - z*z^T) / sigma ** 2
    vectors_product = torch.einsum('ij, ik -> ijk', vectors, vectors)
    eye = torch.eye(dim, device=vectors.device)
    eye = eye.unsqueeze(0)
    eye = eye.repeat(n, 1, 1)
    diff = (eye - vectors_product) / (sigma ** 2)


    s2_vr = score2(samples).reshape(samples.shape[0], -1)
    s2_vr = torch.diag_embed(s2_vr, offset=0, dim1=-2, dim2=-1).reshape(n, dim, dim)
    # with torch.no_grad():
    s1_vr = score1(samples)
    s1_product_vr = torch.einsum('ij, ik -> ijk', s1_vr, s1_vr)
    h_vr = (s2_vr + s1_product_vr).view(n, -1)

    loss = (h_1 ** 2 + h_2 ** 2) + 2 * diff.view(n, -1) * ((h_1 - h_vr) + (h_2 - h_vr))
    loss = loss.sum(dim=-1)
    loss = loss.mean(dim=0) / 2.

    return loss


def dsm_vr(score_net, samples, sigma):
    vectors = torch.randn_like(samples)
    perturbed_inputs = samples + vectors * sigma
    target = vectors / sigma
    perturbed_inputs = perturbed_inputs.view(perturbed_inputs.shape[0], -1)
    scores = score_net(perturbed_inputs)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    vectors = vectors.view(vectors.shape[0], -1)
    loss = ((scores + target) ** 2).sum(dim=-1)

    scores_ = score_net(samples)
    vr = 2. * (scores_ * vectors).sum(dim=-1) / sigma + (vectors ** 2).sum(dim=-1) / (sigma ** 2)
    loss = loss - vr

    loss = loss.mean(dim=0)

    return loss


def dsm(score_net, samples, sigma):
    vectors = torch.randn_like(samples)
    perturbed_inputs = samples + vectors * sigma
    target = vectors / sigma
    perturbed_inputs = perturbed_inputs.view(perturbed_inputs.shape[0], -1)
    scores = score_net(perturbed_inputs)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    loss = ((scores + target) ** 2).sum(dim=-1)

    loss = loss.mean(dim=0)

    return loss