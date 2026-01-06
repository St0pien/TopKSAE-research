import torch

class SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)
        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False, high_precision=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        # store original dtype and work with float64 when high_precision
        original_dtype = r.dtype
        r_work = r.double() if high_precision else r
        
        batch_size, num_dim = r_work.shape
        x = torch.empty_like(r_work, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r_work, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r_work.device)
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
        torch.div(r_work, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)

        torch.where(sign_x, 1 - p, p, out=p)

        # save float64 tensors but mark the original dtype
        ctx.save_for_backward(r_work, x, S)
        ctx.alpha = alpha
        ctx.original_dtype = original_dtype
        ctx.high_precision = high_precision
        
        # return in original dtype if high_precision
        return p.to(original_dtype) if high_precision else p

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha
        original_dtype = ctx.original_dtype
        high_precision = ctx.high_precision

        x = x.clone()
        r = r.clone()
        
        # Work in float64 if high_precision
        grad_output_work = grad_output.double() if high_precision else grad_output

        q_temp = torch.softmax(-torch.abs(x), dim=1)
        qgrad = q_temp * grad_output_work
        grad_k = qgrad.sum(dim=1)
        grad_r = S * q_temp * (grad_k.unsqueeze(1) - grad_output_work)

        # Return in original dtype if high_precision
        return grad_r.to(original_dtype) if high_precision else grad_r, None, None, None, None