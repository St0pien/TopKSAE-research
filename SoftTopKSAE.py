"""
Module for Top-k Sparse SAE (TopKSAE) with Soft Top-K.
"""
import torch
from torch import nn
from overcomplete.sae.base import SAE


class SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)
        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"
        
        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)
            
            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()
            
            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()
            
            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)
            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)
            
            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)
            
            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()
        
        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)
        
        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)
        
        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)
        
        torch.where(sign_x, 1 - p, p, out=p)
        
        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha
        
        # Clone tensors to avoid in-place modifications on saved tensors
        x = x.clone()
        r = r.clone()
        
        q_temp = torch.softmax(-torch.abs(x), dim=1)
        qgrad = q_temp * grad_output
        grad_k = qgrad.sum(dim=1)
        grad_r = S * q_temp * (grad_k.unsqueeze(1) - grad_output)
        
        return grad_r, None, None, None


def soft_top_k(r, k, alpha, descending=False):
    """
    Apply soft top-k selection to input tensor.
    
    Parameters
    ----------
    r : torch.Tensor
        Input tensor of shape (batch_size, num_features).
    k : torch.Tensor or int
        Number of top elements to select. Can be a tensor of shape (batch_size,) 
        or a scalar integer.
    alpha : float
        Temperature parameter controlling the softness of the selection.
        Smaller values make the selection closer to hard top-k.
    descending : bool, optional
        If True, select top-k largest values. If False, select top-k smallest values.
        Default is False.
    
    Returns
    -------
    torch.Tensor
        Soft selection weights of shape (batch_size, num_features).
    """
    if isinstance(k, int):
        k = torch.full((r.shape[0],), k, dtype=torch.long, device=r.device)
    return SoftTopK.apply(r, k, alpha, descending)


class SoftTopKSAE(SAE):
    """
    Soft Top-k Sparse SAE with differentiable top-k selection.
    
    The Soft Top-k Sparse Autoencoder uses a differentiable approximation to
    the top-k operation, allowing gradients to flow through the selection process.
    Instead of hard thresholding, it assigns soft weights to activations.
    
    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
    nb_concepts : int
        Number of components/concepts in the dictionary.
    top_k : int, optional
        Number of top activations to keep, by default nb_concepts // 10.
    alpha : float, optional
        Temperature parameter for soft top-k. Smaller values make it closer
        to hard top-k. Default is 0.1.
    encoder_module : nn.Module or string, optional
        Custom encoder module, by default None.
    dictionary_params : dict, optional
        Parameters for the dictionary layer.
    device : str, optional
        Device to run the model on, by default 'cpu'.
    """
    
    def __init__(self, input_shape, nb_concepts, top_k=None, alpha=0.1,
                 encoder_module=None, dictionary_params=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))
        if isinstance(top_k, int):
            assert top_k > 0
        
        super().__init__(input_shape, nb_concepts, encoder_module,
                         dictionary_params, device)
        
        self.top_k = top_k if top_k is not None else max(nb_concepts // 10, 1)
        self.alpha = alpha
    
    def encode(self, x):
        """
        Encode input data to latent representation with soft top-k.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).
        
        Returns
        -------
        pre_codes : torch.Tensor
            Pre-codes before top-k operation.
        z : torch.Tensor
            Codes with soft top-k selection applied.
        """
        pre_codes, codes = self.encoder(x)
        
        # Soft (differentiable) top-k
        # Get soft weights for top-k largest values
        weights = soft_top_k(codes, self.top_k, self.alpha, descending=True)
        # Apply weights to codes
        z_topk = codes * weights
        
        return pre_codes, z_topk


class TopKSAE(SAE):
    """
    Top-k Sparse SAE with hard (non-differentiable) top-k selection.
    
    This is the original implementation that uses torch.topk for hard selection.
    Use SoftTopKSAE if you want differentiable top-k selection.
    """
    
    def __init__(self, input_shape, nb_concepts, top_k=None,
                 encoder_module=None, dictionary_params=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))
        if isinstance(top_k, int):
            assert top_k > 0
        
        super().__init__(input_shape, nb_concepts, encoder_module,
                         dictionary_params, device)
        
        self.top_k = top_k if top_k is not None else max(nb_concepts // 10, 1)
    
    def encode(self, x):
        """
        Encode input data to latent representation with hard top-k.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).
        
        Returns
        -------
        pre_codes : torch.Tensor
            Pre-codes before top-k operation.
        z : torch.Tensor
            Codes with hard top-k selection applied.
        """
        pre_codes, codes = self.encoder(x)
        
        # Hard (non-differentiable) top-k
        z_topk_result = torch.topk(codes, self.top_k, dim=-1)
        z_topk = torch.zeros_like(codes).scatter(
            -1, z_topk_result.indices, z_topk_result.values
        )
        
        return pre_codes, z_topk
