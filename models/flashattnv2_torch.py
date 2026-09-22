import torch
from torch import einsum

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Q_TILE_SIZE=128, K_TILE_SIZE=64):
        """
        Inputs:
            Q: [B, H, Tq, dk] (Query)
            K: [B, H, Tk, dk] (Key)
            V: [B, H, Tk, dv] (Value)
            is_causal: Whether to apply the causal mask
            Q_TILE_SIZE, K_TILE_SIZE: Tile sizes
        Output:
            O: [B, H, Tq, dv]
        """
        B, H, Tq, dk = Q.shape
        Tk = K.size(2)
        dv = V.size(3)
        scale = 1.0 / (dk ** 0.5)

        # (1) Allocate output tensors.
        O = torch.empty(B, H, Tq, dv, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, H, Tq, device=Q.device, dtype=Q.dtype)

        # (2) Outer loop over Qi.
        for q_start in range(0, Tq, Q_TILE_SIZE):
            # (2.1) load Qi from HBM
            #            Qi ∈ [B, H, Br, dk]
            q_end = min(q_start + Q_TILE_SIZE, Tq)
            Qi = Q[:, :, q_start:q_end, :]

            # (2.2) on chip, initialize O{i,0} ∈ [B, H, Br, dv] =0,
            #                           l{i,0} ∈ [B, H, Br] =0,
            #                           m{i,0} ∈ [B, H, Br] =-♾️
            Oij = torch.zeros(B, H, q_end - q_start, dv, device=Q.device, dtype=Q.dtype)
            lij = torch.zeros(B, H, q_end - q_start, device=Q.device, dtype=Q.dtype)
            mij = torch.full((B, H, q_end - q_start), float('-inf'), device=Q.device, dtype=Q.dtype)

            # (3) Inner loop over Kj and Vj.
            for k_start in range(0, Tk, K_TILE_SIZE):
                # (3.1) load Kj, Vj from HBM
                #            Kj ∈ [B, H, Bc, dk]
                #            Vj ∈ [B, H, Bc, dv]
                k_end = min(k_start + K_TILE_SIZE, Tk)
                Kj = K[:, :, k_start:k_end, :]
                Vj = V[:, :, k_start:k_end, :]

                # (3.2) On chip, compute QK^T: Sij=Qi(Kj)^T ∈ [B, H, Br, Bc].
                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale

                if is_causal: # only correct for Tq == Tk, otherwise need to know causal offset
                    q_pos = torch.arange(q_start, q_end, device=Q.device) # [Br]
                    k_pos = torch.arange(k_start, k_end, device=Q.device) # [Bc]
                    mask = (q_pos[:, None] >= k_pos[None, :]) # [Br, Bc]
                    Sij = Sij.masked_fill(~mask, float('-inf'))

                # (3.3) On chip, compute the row-wise local maximum:
                #                  mij = max(m{i,j-1}, rowmax(Sij)) ∈ [B, H, Br]
                pre_mij = mij
                mij = torch.maximum(mij, torch.max(Sij, dim=-1).values)

                # (3.4) On chip, compute the safe-softmax numerator:
                #                  Pij = exp(Sij - mij) ∈ [B, H, Br, Bc]
                Pij = torch.exp(Sij - mij.unsqueeze(-1))

                # (3.5) On chip, compute the safe-softmax denominator:
                #                  lij = exp(m{i,j-1} - mij)L{i,j-1} + rowsum(Pij) ∈ [B, H, Br]
                lij = torch.exp(pre_mij - mij) * lij + torch.sum(Pij, dim=-1)

                # (3.6) On chip, compute the unnormalized output:
                #                Oij = diag(exp(m{i,j-1} - mij)) Oij + PijVj ∈ [B, H, Br, dv]
                Oij = torch.exp(pre_mij - mij).unsqueeze(-1) * Oij + torch.matmul(Pij, Vj)

            # (2.3) On chip, normalize Oi = diag(lij)^(-1) Oij and write Oi to HBM.
            O[:, :, q_start:q_end, :] =  Oij / lij.unsqueeze(-1)
            # (2.4) On chip, save Li = mij + log(lij) and write Li to HBM.
            #                 logsumexp: exp(Li) recovers the softmax denominator during backward.
            #                 Adding mij lets backward reuse the numerically stable value directly.
            L[:, :, q_start:q_end] = mij + torch.log(lij)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        return O

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx-> Q: [B, H, Tq, dk] (Query)
                  K: [B, H, Tk, dk] (Key)
                  V: [B, H, Tk, dv] (Value)
                  is_causal: Whether to apply the causal mask
                  Q_TILE_SIZE, K_TILE_SIZE: Tile sizes
                  ==================================================
                  O: [B, H, Tq, dv] (output used to compute Di for dPij -> dSij)
                  L: [B, H, Tq] (logsumexp used to reconstruct the softmax denominator)
                  ==================================================
            grad_out: [B, H, Tq, dv] (dO)
        Outputs:
            dQ: [B, H, Tq, dk]
            dK: [B, H, Tk, dk]
            dV: [B, H, Tk, dv]
            None,
            None, None
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE

        B, H, Tq, dk = Q.shape
        Tk = K.size(2)
        dv = V.size(3)
        scale = 1.0 / (dk ** 0.5)

        # (1.1) Allocate gradient tensors; initialize dQ directly in HBM.
        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        # (1.2) Compute Di ∈ [B, H, Tq] for dPij -> dSij and write it to HBM.
        D = torch.sum(O * grad_out, dim=-1)

        # (2) Outer loop over Kj and Vj.
        for k_start in range(0, Tk, K_TILE_SIZE):
            # (2.1) load Kj, Vj from HBM
            #            Kj ∈ [B, H, Bc, dk]
            #            Vj ∈ [B, H, Bc, dv]
            k_end = min(k_start + K_TILE_SIZE, Tk)
            Kj = K[:, :, k_start:k_end, :]
            Vj = V[:, :, k_start:k_end, :]

            # (2.2) initialize dKj ∈ [B, H, Bc, dk] =0,
            #                  dVj ∈ [B, H, Bc, dv] =0
            dKj = torch.zeros_like(Kj)
            dVj = torch.zeros_like(Vj)

            # (3) Inner loop over Qi.
            for q_start in range(0, Tq, Q_TILE_SIZE):
                # (3.1) load Qi, dOi, dQi, Li, Di from HBM
                #            Qi ∈ [B, H, Br, dk]
                #            dOi ∈ [B, H, Br, dv]
                #            dQi ∈ [B, H, Br, dk]
                #            Li ∈ [B, H, Br]
                #            Di ∈ [B, H, Br]
                q_end = min(q_start + Q_TILE_SIZE, Tq)
                Qi = Q[:, :, q_start:q_end, :]

                dOi = grad_out[:, :, q_start:q_end, :]
                dQi = dQ[:, :, q_start:q_end, :]
                Li = L[:, :, q_start:q_end]
                Di = D[:, :, q_start:q_end]

                # (3.2) On chip, recompute QK^T: Sij = Qi(Kj)^T ∈ [B, H, Br, Bc].
                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale

                if is_causal: # only correct for Tq == Tk, otherwise need to know causal offset
                    q_pos = torch.arange(q_start, q_end, device=Q.device) # [Br]
                    k_pos = torch.arange(k_start, k_end, device=Q.device) # [Bc]
                    mask = (q_pos[:, None] >= k_pos[None, :]) # [Br, Bc]
                    Sij = Sij.masked_fill(~mask, float('-inf'))

                # (3.3) On chip, reconstruct Pij using global denominator exp(Li): Pij = exp(Sij - Li).
                Pij = torch.exp(Sij - Li.unsqueeze(-1))

                # (3.4) On chip, compute dPij: dPij = dOi(Vj)^T ∈ [B, H, Br, Bc].
                dPij = torch.matmul(dOi, Vj.transpose(-1, -2))

                # (3.5) On chip, compute dSij: dSij = Pij o (dPij - Di) ∈ [B, H, Br, Bc].
                dSij = Pij * (dPij - Di.unsqueeze(-1)) * scale

                # (3.6) On chip, compute dQi: dQi <- dQi + dSij Kj ∈ [B, H, Br, dk].
                #               dQi is inside the Q-tile loop, so it is loaded from and written to HBM.
                dQi += torch.matmul(dSij, Kj)

                # (3.7) On chip, compute dKj: dKj <- dKj + (dSij)^T Qi ∈ [B, H, Bc, dk].
                #               dKj and dVj remain fixed in the outer loop and can accumulate in SRAM.
                dKj += torch.matmul(dSij.transpose(-1, -2), Qi)

                # (3.8) On chip, compute dVj: dVj <- dVj + (Pij)^T dOi ∈ [B, H, Bc, dv].
                dVj += torch.matmul(Pij.transpose(-1, -2), dOi)

            # (2.3) Write dKj and dVj after the inner loop; SRAM accumulation reduces HBM traffic.
            dK[:, :, k_start:k_end, :] = dKj
            dV[:, :, k_start:k_end, :] = dVj

        return dQ, dK, dV, None, None, None


def attention(q, k, v, is_causal=False):
    """Run the manual PyTorch implementation with its published tile sizes."""
    return FlashAttentionTorch.apply(q, k, v, is_causal, 128, 64)


__all__ = ["FlashAttentionTorch", "attention"]
