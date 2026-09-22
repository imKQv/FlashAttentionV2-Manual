import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Generic fallback.
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4, num_stages=2),
        # Suitable for SM86/SM89 devices with limited shared memory.
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=8, num_stages=3),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=2),
        # Candidates for devices with more shared memory, such as A100/H100.
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8, num_stages=1),
    ],
    key=["N_QUERIES", "N_KEYS", "D", "is_causal", "INPUT_DTYPE"],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, INPUT_DTYPE: tl.constexpr
):
    # (2) Outer loop over Qi.
    batch_idx = tl.program_id(2)
    head_idx  = tl.program_id(1)
    query_tile_idx = tl.program_id(0)

    # (2.1) load Qi from HBM
    #            Qi ∈ [B, H, Br, dk]
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb + head_idx * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb + head_idx * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb + head_idx * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # (2.2) on chip, initialize O{i,0} ∈ [B, H, Br, dv] =0,
    #                           l{i,0} ∈ [B, H, Br] =0,
    #                           m{i,0} ∈ [B, H, Br] =-♾️
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    Li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    """
    Sequence-tail handling differs from the Torch implementation:
    Torch slices Qi to its actual length, but Triton block_shape values are
    compile-time constants, so the final Query tile keeps Q_TILE_SIZE rows.
    In causal mode, causal_mask covers future Key columns; in non-causal mode,
    k_mask removes padded Key columns. boundary_check prevents padded Query
    rows from being stored in both modes. Do not mask those Query rows here:
    all -inf rows would cause an unstable -inf - (-inf) online-softmax step.
    """
    q_start = query_tile_idx * Q_TILE_SIZE
    q_idx = q_start + tl.arange(0, Q_TILE_SIZE) # [Br]

    # (3) Inner loop over Kj and Vj.
    for key_tile_idx in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # (3.1) load Kj, Vj from HBM
        #            Kj ∈ [B, H, Bc, dk]
        #            Vj ∈ [B, H, Bc, dv]
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # (3.2) On chip, compute QK^T: Sij=Qi(Kj)^T ∈ [B, H, Br, Bc].
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # Matching FP16/BF16/FP32 inputs produce FP32 output.

        """
        Key/Value tail tiles retain a fixed K_TILE_SIZE. boundary_check pads
        out-of-bounds loads with zero, but without k_mask those zeros become valid
        score=0 entries in non-causal softmax. Set padded Key scores to -inf.
        """
        k_start = key_tile_idx * K_TILE_SIZE
        k_idx = k_start + tl.arange(0, K_TILE_SIZE)
        k_mask = k_idx < N_KEYS

        if is_causal: # only correct for Tq == Tk, otherwise need to know causal offset
            causal_mask = q_idx[:, None] >= k_idx[None, :] # [Br, Bc]
            final_mask = k_mask[None, :] & causal_mask
        else:
            final_mask = k_mask[None, :]

        Sij = tl.where(final_mask, Sij, float("-inf"))

        # (3.3) On chip, compute the row-wise local maximum:
        #                  mij = max(m{i,j-1}, rowmax(Sij)) ∈ [B, H, Br]
        pre_mi = mi
        mi = tl.maximum(mi, tl.max(Sij, axis=1))

        # (3.4) On chip, compute the safe-softmax numerator:
        #                  Pij = exp(Sij - mij) ∈ [B, H, Br, Bc]
        Pij = tl.exp(Sij - mi[:, None])

        # (3.5) On chip, compute the safe-softmax denominator:
        #                  lij = exp(m{i,j-1} - mij)L{i,j-1} + rowsum(Pij) ∈ [B, H, Br]
        Li = tl.exp(pre_mi - mi) * Li + tl.sum(Pij, axis=1)

        # (3.6) On chip, compute the unnormalized output:
        #                Oij = diag(exp(m{i,j-1} - mij)) Oij + PijVj ∈ [B, H, Br, dv]
        Oi = tl.exp(pre_mi - mi)[:, None] * Oi
        Oi = tl.dot(Pij.to(Vj.dtype), Vj, acc=Oi)  # Matching low-precision inputs accumulate into FP32 Oi.

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # (2.3) On chip, normalize Oi = diag(lij)^(-1) Oij and write Oi to HBM.
    Oi = Oi / Li[:, None]
    # (2.4) On chip, save Li = mij + log(lij) and write Li to HBM.
    #                 logsumexp: exp(Li) recovers the softmax denominator during backward.
    #                 Adding mij lets backward reuse the numerically stable value directly.
    Li = mi + tl.log(Li)

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_idx * stride_ob + head_idx * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb + head_idx * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    # Oi stays FP32 in SRAM; cast explicitly because block-pointer stores require the destination element type.
    # boundary_check uses the block-pointer shape to prevent padded rows from being stored or loaded.
    tl.store(O_block_ptr, Oi.to(Qi.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))



@triton.autotune(
    configs=[
        # Generic fallback.
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4, num_stages=2),
        # Suitable for SM86/SM89 devices with limited shared memory.
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=8, num_stages=3),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=1),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=2),
        # Candidates for devices with more shared memory, such as A100/H100.
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8, num_stages=1),
        triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8, num_stages=2),
        triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8, num_stages=1),
    ],
    key=["N_QUERIES", "N_KEYS", "D", "is_causal", "INPUT_DTYPE"],
    reset_to_zero=["dQ_ptr"],
)
@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    stride_dob, stride_doh, stride_doq, stride_dod,
    stride_db, stride_dh, stride_dq,
    stride_dqb, stride_dqh, stride_dqq, stride_dqd,
    stride_dkb, stride_dkh, stride_dkk, stride_dkd,
    stride_dvb, stride_dvh, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, INPUT_DTYPE: tl.constexpr
):
    # (2) Outer loop over Kj and Vj.
    batch_idx = tl.program_id(2)
    head_idx  = tl.program_id(1)
    key_tile_idx = tl.program_id(0)

    # (2.1) load Kj, Vj from HBM
    #            Kj ∈ [B, H, Bc, dk]
    #            Vj ∈ [B, H, Bc, dv]
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb + head_idx * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb + head_idx * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # (2.2) initialize dKj ∈ [B, H, Bc, dk] =0,
    #                  dVj ∈ [B, H, Bc, dv] =0
    dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # Accumulate across Query tiles in FP32.
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)  # Accumulate across Query tiles in FP32.

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb + head_idx * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_idx * stride_dob + head_idx * stride_doh,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb + head_idx * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_idx * stride_db + head_idx * stride_dh,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dQ_base = dQ_ptr + batch_idx * stride_dqb + head_idx * stride_dqh

    # (3) Inner loop over Qi.
    for query_tile_idx in range(0, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # (3.1) load Qi, dOi, dQi, Li, Di from HBM
        #            Qi ∈ [B, H, Br, dk]
        #            dOi ∈ [B, H, Br, dv]
        #            dQi ∈ [B, H, Br, dk]
        #            Li ∈ [B, H, Br]
        #            Di ∈ [B, H, Br]
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # Internal FP32 buffer.
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")  # Internal FP32 buffer.

        # (3.2) On chip, recompute QK^T: Sij = Qi(Kj)^T ∈ [B, H, Br, Bc].
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # Matching FP16/BF16/FP32 inputs produce FP32 output.

        """
        Backward reconstruction must also handle padded sequence tails explicitly.
        Torch slices tensors to their actual tail length, while Triton still computes
        a fixed Q_TILE_SIZE x K_TILE_SIZE block. q_mask and k_mask exclude padded
        Query and Key positions from Pij and gradient accumulation. Unlike forward,
        backward must combine q_mask[:, None] and k_mask[None, :] in both causal and
        non-causal modes. Li is finite for valid rows, so masking padded rows to -inf
        does not introduce the forward online-softmax -inf - (-inf) instability.
        """
        q_start = query_tile_idx * Q_TILE_SIZE
        q_end = tl.minimum(q_start + Q_TILE_SIZE, N_QUERIES)
        q_range = q_end - q_start
        q_idx = q_start + tl.arange(0, Q_TILE_SIZE) # [Br]
        q_mask = tl.arange(0, Q_TILE_SIZE) < q_range

        k_start = key_tile_idx * K_TILE_SIZE
        k_end = tl.minimum(k_start + K_TILE_SIZE, N_KEYS)
        k_range = k_end - k_start
        k_idx = k_start + tl.arange(0, K_TILE_SIZE) # [Bc]
        k_mask = tl.arange(0, K_TILE_SIZE) < k_range

        valid_mask = q_mask[:, None] & k_mask[None, :]

        if is_causal: # only correct for Tq == Tk, otherwise need to know causal offset
            causal_mask = q_idx[:, None] >= k_idx[None, :] # [Br, Bc]
            final_mask = valid_mask & causal_mask
        else:
            final_mask = valid_mask

        Sij = tl.where(final_mask, Sij, float('-inf'))

        # (3.3) On chip, reconstruct Pij using global denominator exp(Li): Pij = exp(Sij - Li).
        Pij = tl.exp(Sij - Li[:, None])

        # (3.4) On chip, compute dPij: dPij = dOi(Vj)^T ∈ [B, H, Br, Bc].
        dPij = tl.dot(dOi, tl.trans(Vj))  # Matching FP16/BF16/FP32 inputs produce FP32 output.

        # (3.5) On chip, compute dSij: dSij = Pij o (dPij - Di) ∈ [B, H, Br, Bc].
        dSij = Pij * (dPij - Di[:, None]) * scale

        # (3.6) On chip, compute dQi: dQi <- dQi + dSij Kj ∈ [B, H, Br, dk].
        #               dQi is inside the Q-tile loop, so each iteration reads and writes HBM.
        dQi = tl.dot(dSij.to(Kj.dtype), Kj)  # Matching low-precision inputs produce FP32 output.

        """
        Each program owns one K tile, and programs for different K tiles may run
        concurrently. Their partial dQ values require a reduction, so this kernel
        uses atomic_add. Each program traverses all Q tiles for its owned K tile,
        allowing dKj and dVj to accumulate in SRAM without competing writes.
        dQ atomic contention is a major backward bottleneck. A split dQ kernel
        would instead assign one Q tile to each program and traverse all K tiles.
        """
        row_offsets = (query_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None]
        col_offsets = tl.arange(0, D)[None, :]
        ptrs = dQ_base + row_offsets * stride_dqq + col_offsets * stride_dqd

        tl.atomic_add(ptrs, dQi, mask=q_mask[:, None])

        # (3.7) On chip, compute dKj: dKj <- dKj + (dSij)^T Qi ∈ [B, H, Bc, dk].
        #               dKj and dVj remain fixed in the outer loop and accumulate in SRAM.
        dKj = tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi, acc=dKj)  # FP32 accumulation.

        # (3.8) On chip, compute dVj: dVj <- dVj + (Pij)^T dOi ∈ [B, H, Bc, dv].
        dVj = tl.dot(tl.trans(Pij.to(dOi.dtype)), dOi, acc=dVj)  # FP32 accumulation.

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    # (2.3) Write dKj and dVj after FP32 accumulation in SRAM to reduce HBM traffic.
    # Block-pointer stores do not implicitly cast FP32 to low precision; cast to dK/dV element types.
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_idx * stride_dkb + head_idx * stride_dkh,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_idx * stride_dvb + head_idx * stride_dvh,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    tl.store(dK_block_ptr, dKj.to(Kj.dtype), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj.to(Vj.dtype), boundary_check=(0, 1))



class FlashAttenTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        """
        Q: [B, H, Tq, dk]
        K: [B, H, Tk, dk]
        V: [B, H, Tk, dv]
        is_causal: causal mode currently requires Tq == Tk
        """
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError("Q, K, and V must be four-dimensional [B, H, T, D] tensors")
        if Q.device.type != "cuda" or K.device != Q.device or V.device != Q.device:
            raise ValueError("Q, K, and V must be on the same CUDA device")
        if Q.dtype != K.dtype or Q.dtype != V.dtype:
            raise ValueError("Q, K, and V must have the same dtype")
        if Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("The Triton kernel supports only FP16, BF16, and FP32")

        B, H, Tq, dk = Q.shape
        Tk = K.shape[2]
        dv = V.shape[3]
        if K.shape[:2] != (B, H) or V.shape[:2] != (B, H):
            raise ValueError("Q, K, and V must have matching batch and head dimensions")
        if K.shape[3] != dk:
            raise ValueError("Q and K must have the same head dimension")
        if V.shape[2] != Tk:
            raise ValueError("K and V must have the same sequence length")
        if dv != dk:
            raise ValueError("The Triton kernel requires dk == dv")
        if dk not in (16, 32, 64, 128):
            raise ValueError(
                "The Triton kernel supports head dimension D in {16, 32, 64, 128}"
            )
        if is_causal and Tq != Tk:
            raise ValueError("causal=True currently requires Tq == Tk")
        scale = 1.0 / (dk ** 0.5)
        input_dtype = {
            torch.float16: 0,
            torch.bfloat16: 1,
            torch.float32: 2,
        }[Q.dtype]

        # (1) Allocate output tensors.
        # L is an internal FP32 buffer.
        O = torch.empty(B, H, Tq, dv, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, H, Tq, device=Q.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(Tq, META["Q_TILE_SIZE"]),
            H,
            B,
        )

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            Tq, Tk, scale,
            D=dk, is_causal=is_causal, INPUT_DTYPE=input_dtype
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        B, H, Tq, dk = Q.shape
        Tk = K.shape[2]
        dv = V.shape[3]

        scale = 1.0 / (dk ** 0.5)
        input_dtype = {
            torch.float16: 0,
            torch.bfloat16: 1,
            torch.float32: 2,
        }[Q.dtype]

        # (1.2) Compute Di ∈ [B, H, Tq] for dPij -> dSij and write it to HBM.
        # D is an internal FP32 buffer.
        D = torch.sum(grad_out.to(torch.float32) * O.to(torch.float32), dim=-1)

        # (1.1) Allocate gradient tensors; initialize dQ directly in HBM.
        # Keep the global dQ workspace in FP32 and cast once on return to avoid repeated HBM conversions.
        # The official FlashAttentionV2 CUDA implementation also uses an FP32 global workspace.
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        grid = lambda META: (
            triton.cdiv(Tk, META["K_TILE_SIZE"]),
            H,
            B,
        )

        flash_bwd_kernel[grid](
            Q, K, V, O, L, grad_out, D, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            Tq, Tk, scale,
            D=dk, is_causal=is_causal, INPUT_DTYPE=input_dtype
        )

        # Cast the FP32 global workspace back to Q.dtype on return.
        return dQ.to(Q.dtype), dK, dV, None


def attention(q, k, v, is_causal=False):
    """Run the manual Triton implementation."""
    return FlashAttenTriton.apply(q, k, v, is_causal)


__all__ = ["FlashAttenTriton", "attention"]
