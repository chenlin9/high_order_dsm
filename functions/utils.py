import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def get_jvp(y, x, v):
    '''
    Generate jacobian vector product. Requires x.requires_grad() and v.requires_grad(). Refer to `https://gist.github.com/ybj14/7738b119768af2fe765a2d63688f5496`.
    Args:
        y: [batch_size, n_output]
        x: [batch_size, n_input]
        v: [n_input, batch_size]
    Returns:
        jvp: [n_output, batch_size]
    '''
    print(y)
    print(x)
    print(v)
    u = torch.zeros_like(y, requires_grad=True)
    ujp = torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)
    ujpT = ujp[0].transpose(1, 0)
    jvpT = torch.autograd.grad(ujpT, u, grad_outputs=v, retain_graph=True)
    jvp = jvpT[0].transpose(1, 0)

    return jvp


def get_jacobian(net, x):
    '''
    Generate jacobian. Requires x.requires_grad(). Refer to `https://gist.github.com/ybj14/7738b119768af2fe765a2d63688f5496`.
    Args:
        net: pytorch model, [batch_size, n_input] -> [batch_size, n_output]
        x: [batch_size, n_input]
    Returns:
        j: [n_batch, n_output, n_in]
    '''
    n_batch = x.shape[0]
    n_in = x.shape[1]
    n_output = net(x).shape[1]
    # jacobians = []
    #  x: [n_batch, n_in]
    # xs: [n_batch * n_output, n_in]
    xs = x.repeat(1, n_output).view(-1, n_in)
    xs_grad = None

    def hook(grad):
        nonlocal xs_grad
        xs_grad = grad

    xs.register_hook(hook)
    y = net(xs)
    y.backward(torch.eye(n_output, device=x.device).repeat(n_batch, 1), retain_graph=True)
    # xs.grad.data: [n_batch * n_output, n_in]
    return xs_grad.view(n_batch, n_output, n_in)


def symsqrt(matrix):
    """
    Compute the square root of a batch of positive definite matrix. Refer to `https://github.com/pytorch/pytorch/issues/25481`.
    Args:
        matrix: [batch_size, n_input, n_input]
    Returns:
        sqrt: [batch_size, n_input, n_input]
    """
    # device = matrix.device
    # matrix = matrix.to("cpu")
    try:
        _, s, v = torch.svd(matrix)
    except:
        # import pdb
        # pdb.set_trace()
        _, s, v = torch.svd(matrix + 1e-4 * matrix.mean() * torch.rand_like(matrix))

    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))

    output = (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)
    # output = output.to(device)
    return output


def torch_expm(A):
    """
    Refer to `https://github.com/SkafteNicki/cuda_expm/blob/master/expm.py`.
    """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))

    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings
    n_squarings = n_squarings.flatten().type(torch.int32)

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q)  # solve P = Q*R

    # Unsquaring step
    expmA = []
    for i in range(n_A):
        l = [R[i]]
        for _ in range(n_squarings[i]):
            l.append(l[-1].mm(l[-1]))
        expmA.append(l[-1])
    return torch.stack(expmA)


def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)


def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V
