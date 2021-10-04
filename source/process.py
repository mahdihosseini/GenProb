from __future__ import division
from contextlib import contextmanager
import math
from torch import nn, optim
import torchvision
import sys
import random
import numpy.linalg as LA
import torch
import time
import numpy as np
from nats_bench import create
from copy import deepcopy
import torchvision.transforms as transforms
from scipy.optimize import minimize_scalar
from pprint import pprint


def EVBMF(Y, sigma2=None, H=None):
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    # U, s, V = np.linalg.svd(Y)
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.
    if H < L:
        # residual = np.sum(np.sum(Y**2)-np.sum(s**2))
        residual = torch.sum(np.sum(Y**2)-np.sum(s**2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1+tauubar)*(1+alpha/tauubar)
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        upper_bound = (torch.sum(s**2)+residual)/(L*M)
        lower_bound = torch.max(torch.stack(
            [s[eH_ub+1]**2/(M*xubar), torch.mean(s[eH_ub+1:]**2)/M], dim=0))

        scale = 1.  # /lower_bound
        s = s*np.sqrt(scale)
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale

        sigma2_opt = minimize_scalar(
            EVBsigma2, args=(L, M, s.cpu().numpy(), residual, xubar),
            bounds=[lower_bound.cpu().numpy(), upper_bound.cpu().numpy()],
            method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    pos = torch.sum(s > threshold)
    d = (s[:pos]/2)*(1-(L+M)*sigma2/s[:pos]**2 +
                     torch.sqrt((1 -
                                 (L+M)*sigma2/s[:pos]**2)**2 - 4*L*M*sigma2**2/s[:pos]**4))

    return U[:, :pos], torch.diag(d), V[:, :pos]  # , post


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)

    alpha = L/M
    x = s**2/(M*sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1+1, z1)))
    term4 = alpha*np.sum(np.log(tau_z1/alpha+1))

    obj = term1+term2+term3+term4 + residual/(M*sigma2) + (L-H)*np.log(sigma2)

    return obj


def phi0(x):
    return x-np.log(x)


def phi1(x, alpha):
    return np.log(tau(x, alpha)+1) + alpha*np.log(tau(x, alpha)/alpha + 1
                                                  ) - tau(x, alpha)


def tau(x, alpha):
    return 0.5 * (x-(1+alpha) + np.sqrt((x-(1+alpha))**2 - 4*alpha))

def compute_low_rank(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.requires_grad:
        tensor = tensor.detach()
    try:
        tensor_size = tensor.shape
        if tensor_size[0] > tensor_size[1]:
            tensor = tensor.T
            tensor_size = tensor.shape
        U_approx, S_approx, V_approx = EVBMF(tensor)
    except RuntimeError as error:
        print(error)
        return None, None, None
    rank = S_approx.shape[0] / tensor_size[0]
    low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
    #print(tensor)
    if len(low_rank_eigen) != 0:
        condition = low_rank_eigen[0] / low_rank_eigen[-1]

        effective_rank = low_rank_eigen/np.sum(low_rank_eigen)
        effective_rank_ln = np.log(effective_rank)
        effective_rank = np.multiply(effective_rank,effective_rank_ln)
        effective_rank = -np.sum(effective_rank)

        sum_low_rank_eigen = low_rank_eigen/max(low_rank_eigen)
        sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
        KG = sum_low_rank_eigen / tensor_size[0]
    else:
        condition = 0
        effective_rank = 0
        KG = 0
    return [KG, condition, effective_rank]

def compute(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.requires_grad:
        tensor = tensor.detach()
    try:
        tensor_size = tensor.shape
        if tensor_size[0] > tensor_size[1]:
            tensor = tensor.T
            tensor_size = tensor.shape
        U, S, V = torch.svd(tensor)
    except RuntimeError:
        return None, None, None
    low_rank_eigen = S.data.cpu().numpy()
    if len(low_rank_eigen) != 0:
        condition = low_rank_eigen[0] / low_rank_eigen[-1]

        effective_rank = low_rank_eigen/np.sum(low_rank_eigen)
        effective_rank_ln = np.log(effective_rank)
        effective_rank = np.multiply(effective_rank,effective_rank_ln)
        effective_rank = -np.sum(effective_rank)

        sum_low_rank_eigen = low_rank_eigen/max(low_rank_eigen)
        sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
        KG = sum_low_rank_eigen / tensor_size[0]
    else:
        condition = 0
        effective_rank = 0
        KG = 0

    return [KG, condition, effective_rank]

def norms_low_rank(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    try:
        tensor_size = tensor.shape
        if tensor_size[0] > tensor_size[1]:
            tensor = tensor.T
            tensor_size = tensor.shape
        U_approx, S_approx, V_approx = EVBMF(tensor)
    except RuntimeError as error:
        print(error)
        return None, None, None
    low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
    if(len(low_rank_eigen)>0):
        spec_norm = max(low_rank_eigen)
    else:
        spec_norm = 0
    low_rank_tensor = np.dot(np.dot(U_approx,S_approx),V_approx.T)
    fro_norm = LA.norm(low_rank_tensor,ord='fro')
    return [spec_norm, fro_norm]

def norms(tensor):
    #spec_norm = LA.norm(tensor,ord=2)
    U, s, V = torch.svd(tensor)
    spec_norm = max(s)
    fro_norm = LA.norm(tensor,ord='fro')
    return [spec_norm, fro_norm]

class Welford:

    def __init__(self):
        self.k = torch.tensor([0]).cuda()

    def update(self, newValue):
        if(self.k==0):
            self.M = torch.zeros(len(newValue)).cuda()
            self.m = torch.zeros(len(newValue)).cuda()
            self.S = torch.zeros(len(newValue)).cuda()
        self.k += 1
        delta = newValue - self.m
        self.m += delta / self.k
        delta2 = newValue - self.m
        self.M += delta * delta2

    def finalize(self):
        if self.k < 2:
            return float("nan")
        else:
            (mean2, variance, sampleVariance) = ((self.m**2).cpu(), (self.M / self.k).cpu(), (self.M / (self.k - 1)).cpu())
            return (mean2, variance, sampleVariance)

@contextmanager
def _perturbed_model(
  model,
  sigma: float = 1,
  rng = torch.Generator(),
  magnitude_eps = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model     

@torch.no_grad()
def _pacbayes_sigma(
  model,
  dataloader,
  accuracy: float,
  seed: int,
  magnitude_eps = None,
  search_depth: int = 4,
  montecarlo_samples: int = 10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
  rng.manual_seed(BIG_NUMBER + seed)

  for __ in range(search_depth):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(montecarlo_samples):
      with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            logits = p_model(data)
            pred = (logits[1]).data.max(1, keepdim=True)[1]  # get the index of the max logits
            batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
            loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
        print("[",__,_, loss_estimate, "]")
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

def get_dataset_dep(model, dataset, margin_param, GSNR_params, pac_params):
    '''
    if "cifar10" in dataset or "CIFAR10" in dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers = 0)
    for data, target in dataloader:
        shape = data.shape[1:]
        break
    '''
    shape = (3, 32, 32)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device)
    '''
    if(margin_param):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers = 0)
        hit = torch.tensor([0])
        margins = []
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = np.asarray(model(data))[1]
            hit += torch.sum(torch.argmax(logits,axis=1) == target).cpu()
            correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
            logits[torch.arange(logits.shape[0]), target] = float('-inf')
            max_other_logit = logits.data.max(1).values  # get the index of the max logits
            margin = correct_logit - max_other_logit
            margin = margin.clone().detach().cpu()
            margins.append(margin)
        margin = torch.cat(margins).kthvalue(m // 10)[0]
        acc = hit/m
    else:
        margin =0
    '''
    #path norm
    model1 = deepcopy(model)
    model1.eval()
    for param in model1.parameters():
      if param.requires_grad:
        param.data.pow_(2)
    expand = [1]
    expand.extend(shape)
    x = torch.ones(expand)
    x = model1(x)
    del model1
    try:
        x = x[1].clone().detach()
    except:
        x = x.clone().detach()
    pathnorm = math.sqrt(torch.sum(x))
    '''
    #gsnr
    if(GSNR_params[0]):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 0)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        model.eval()

        child = Welford()
        i=0
        for data, target in dataloader:

            optimizer.zero_grad()

            outputs = model(data.cuda())
            loss = criterion(outputs[1], target.cuda())
            loss.backward() #Gradients calculated
            grad_history = []

            for param in model.parameters():
                if(param.requires_grad):
                    grad_history.append(param.grad.flatten())
            grad_history = torch.cat(grad_history)

            child.update(grad_history)

            del grad_history

            if(i==GSNR_params[1]):
                break

            if(i%1000==0):
                print(i)
            i+=1
        
        mean2, var, svar = child.finalize()
        del(child)
        mean2 = mean2[mean2!=0]
        svar = svar[svar!=0]
        gsnr = mean2/svar
        gsnr = torch.mean(gsnr)

    else:
        gsnr = 0


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers = 0)
    #sigma
    if pac_params[0][0] == 1:
        seed = 0
        pac_sigma = _pacbayes_sigma(model, dataloader, acc, seed, search_depth=pac_params[1])
    else:
        pac_sigma = 0
    #pac sigma
    if(pac_params[0][1]==1):
        seed = 0
        mag_eps = 1e-3
        mag_pac_sigma = _pacbayes_sigma(model, dataloader, acc, seed, magnitude_eps=mag_eps, search_depth=pac_params[1])
    else:
        mag_pac_sigma = 0
    '''
    model = model.cpu()
    return np.asarray([pathnorm])


def get_metrics(weight):
    layer_tensor=weight
    tensor_size = layer_tensor.shape

    in_metrics_BE = []
    out_metrics_BE = []
    in_metrics_AE = []
    out_metrics_AE = []

    type = 0

    if (len(tensor_size)==4):
        mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
        mode_3_unfold = torch.reshape(mode_3_unfold, [tensor_size[1], tensor_size[0]*tensor_size[2]*tensor_size[3]])

        in_metrics_AE.extend(compute_low_rank(mode_3_unfold))
        in_metrics_AE.extend(norms_low_rank(mode_3_unfold))
        in_weight_AE = min(tensor_size[1],tensor_size[0] * tensor_size[2] * tensor_size[3])

        in_metrics_BE.extend(compute(mode_3_unfold))
        in_metrics_BE.extend(norms(mode_3_unfold))
        in_weight_BE = min(tensor_size[1],tensor_size[0] * tensor_size[2] * tensor_size[3])

        mode_4_unfold = layer_tensor
        mode_4_unfold = torch.reshape(mode_4_unfold, [tensor_size[0], tensor_size[1]*tensor_size[2]*tensor_size[3]])

        out_metrics_AE.extend(compute_low_rank(mode_4_unfold))
        out_metrics_AE.extend(norms_low_rank(mode_4_unfold))
        out_weight_AE = min(tensor_size[0],tensor_size[1] * tensor_size[2] * tensor_size[3])

        out_metrics_BE.extend(compute(mode_4_unfold))
        out_metrics_BE.extend(norms(mode_4_unfold))
        out_weight_BE = min(tensor_size[0],tensor_size[1] * tensor_size[2] * tensor_size[3])

        type = 4
    elif (len(tensor_size)==2):
        in_metrics_AE.extend(compute_low_rank(layer_tensor))
        in_metrics_AE.extend(norms_low_rank(layer_tensor))
        in_weight_AE = min(tensor_size[1],tensor_size[0])

        in_metrics_BE.extend(compute(layer_tensor))
        in_metrics_BE.extend(norms(layer_tensor))
        in_weight_BE = min(tensor_size[1],tensor_size[0])

        out_metrics_AE.extend(compute_low_rank(layer_tensor))
        out_metrics_AE.extend(norms_low_rank(layer_tensor))
        out_weight_AE = in_weight_AE

        out_metrics_BE.extend(compute(layer_tensor))
        out_metrics_BE.extend(norms(layer_tensor))
        out_weight_BE = in_weight_BE

        type = 2
    else:
        return None

    return np.concatenate((in_metrics_BE,out_metrics_BE,in_metrics_AE,out_metrics_AE)), [in_weight_BE, out_weight_BE, in_weight_AE, out_weight_AE], type


