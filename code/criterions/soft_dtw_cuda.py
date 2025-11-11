# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
# Modified 2025 by ChatGPT (fix: CUDA thread-limit safety + clearer fallbacks)

from numba import config
config.CUDA_DEFAULT_PTX_CC = (8, 0)

import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math
import os

# ----------------------------------------------------------------------------------------------------------------------
# CUDA kernels (stride-based across I indices). 
# NOTE: kernels now accept `start_pass` and `n_chunk_passes` so host can
#       split the full n_passes into safe-sized chunks to avoid long-running kernels.
# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, start_pass, n_chunk_passes, R):
    """
    CUDA diagonal implementation (stride-based) for a chunk of anti-diagonals.
    - D: (B, N, M) float32
    - R: (B, N+2, M+2) float32
    - start_pass: starting diagonal index (0-based)
    - n_chunk_passes: number of diagonals to process in this launch (<= chunk cap)
    """
    b = cuda.blockIdx.x  # one block per batch item
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    inv_gamma = 1.0 / gamma

    # local variables to minimize global loads
    max_i_local = max_i
    max_j_local = max_j

    for p_local in range(n_chunk_passes):
        p = start_pass + p_local
        I = tid
        while I < max_i_local:
            J = p - I
            if 0 <= J < max_j_local:
                i = I + 1
                j = J + 1
                # bandwidth check: skip if difference exceeds bandwidth and bandwidth>0
                if not (abs(i - j) > bandwidth > 0):
                    # careful indexing: R has padding of +1 on each side; D indexed [i-1, j-1]
                    r0 = -R[b, i - 1, j - 1] * inv_gamma
                    r1 = -R[b, i - 1, j] * inv_gamma
                    r2 = -R[b, i, j - 1] * inv_gamma
                    # numerically stable soft-min
                    rmax = max(max(r0, r1), r2)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin
            I += stride
        # synchronize threads in block for this diagonal step
        cuda.syncthreads()


@cuda.jit
def compute_softdtw_backward_cuda(D_, R, inv_gamma, bandwidth, max_i, max_j, start_pass, n_chunk_passes, E):
    """
    Backward pass on CUDA for a chunk of anti-diagonals in reverse.
    - D_ has padding (B, N+2, M+2)
    - R is the forward R (B, N+2, M+2)
    - E is gradient accumulator (B, N+2, M+2)
    """
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x

    # We'll process chunk of anti-diagonals in reverse order:
    # start_pass is index into full reversed passes; the host must compute correct reversed indices.
    for p_local in range(n_chunk_passes):
        # rev_p is counted from 0 for the reversed passes chunk
        rev_p = start_pass + p_local  # host computes proper reversed start
        I = tid
        while I < max_i:
            J = rev_p - I
            if 0 <= J < max_j:
                i = I + 1
                j = J + 1

                if math.isinf(R[k, i, j]):
                    # keep same semantic as original: convert +inf to -inf for backward logic
                    R[k, i, j] = -math.inf

                if not (abs(i - j) > bandwidth > 0):
                    # Use R and D_ with safe indexing (D_ already padded)
                    a = math.exp((R[k, i + 1, j] - R[k, i, j] - D_[k, i + 1, j]) * inv_gamma)
                    b = math.exp((R[k, i, j + 1] - R[k, i, j] - D_[k, i, j + 1]) * inv_gamma)
                    c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D_[k, i + 1, j + 1]) * inv_gamma)
                    E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
            I += stride
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
# Safe SoftDTW autograd Function using chunked kernel launches and explicit device selection
# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA-backed autograd Function with chunked launches to avoid long-running kernels/timeouts.
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        if not D.is_cuda:
            raise RuntimeError("D must be a CUDA tensor for _SoftDTWCUDA")

        dev = D.device
        dtype = D.dtype
        gamma_val = float(gamma)
        bandwidth_val = float(bandwidth)

        B = int(D.shape[0])
        N = int(D.shape[1])
        M = int(D.shape[2])

        # determine device index robustly
        if isinstance(dev, torch.device):
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        else:
            dev_idx = torch.cuda.current_device()

        # ensure numba uses same device context
        try:
            cuda.select_device(dev_idx)
        except Exception:
            # fallback: try to continue, but this select_device is important for driver->devptr mapping
            pass

        # get device properties and cap threads per block
        try:
            props = torch.cuda.get_device_properties(dev_idx)
            max_threads = props.max_threads_per_block
        except Exception:
            max_threads = 1024

        SAFE_THREAD_CAP = min(max_threads, 256)
        threads_per_block = int(min(max(N, M), SAFE_THREAD_CAP)) if SAFE_THREAD_CAP > 0 else 1
        if threads_per_block <= 0:
            threads_per_block = 1

        # grid (one block per batch item). Numba supports up to large grid dims but be conservative.
        blocks_per_grid = B

        # number of anti-diagonals
        n_passes = N + M - 1

        # chunk size: limit the number of anti-diagonals processed per launch to avoid long-running kernels.
        # empirical safe chunk size (tunable). Keep <= 1024 typically.
        CHUNK_SIZE = 1024
        # If sequences are extremely long, reduce chunk size further
        if max(N, M) > 512:
            CHUNK_SIZE = 512

        # Prepare R buffer (float32 for CUDA kernel)
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=torch.float32) * math.inf
        R[:, 0, 0] = 0.0

        # Convert D to float32 for kernel (kernel expects float32 arrays)
        D_cuda = D.detach().to(torch.float32)

        if max(N, M) > SAFE_THREAD_CAP:
            print(f"[SoftDTW] Note: sequence length {max(N, M)} > safe thread cap {SAFE_THREAD_CAP}. "
                  "Kernel will use stride-based coverage across diagonals (safe).")

        # Launch forward in chunks
        start = 0
        while start < n_passes:
            n_chunk = min(CHUNK_SIZE, n_passes - start)
            # pass start_pass and n_chunk_passes
            # ensure device context is active for numba to get device pointer
            try:
                cuda.select_device(dev_idx)
                compute_softdtw_cuda[blocks_per_grid, threads_per_block](
                    cuda.as_cuda_array(D_cuda),
                    gamma_val, float(bandwidth_val), N, M, start, n_chunk,
                    cuda.as_cuda_array(R)
                )
            except Exception as e:
                # if as_cuda_array mapping fails, try with a pinned fallback by copying to numba device array
                # but copying is expensive: fallback to CPU path
                print("[SoftDTW] CUDA forward failed with exception:", repr(e))
                print("[SoftDTW] Falling back to CPU implementation for forward.")
                R_cpu = compute_softdtw(D_cuda.cpu().numpy(), gamma_val, bandwidth_val)
                R = torch.tensor(R_cpu, device=dev).type(torch.float32)
                break
            start += n_chunk

        ctx.save_for_backward(D, R.clone(), torch.tensor(gamma_val, device=dev), torch.tensor(bandwidth_val, device=dev))
        result = R[:, -2, -2]
        if dtype != torch.float32:
            result = result.to(dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma_t, bandwidth_t = ctx.saved_tensors
        dev = grad_output.device
        dtype = grad_output.dtype
        gamma_val = float(gamma_t.item())
        bandwidth_val = float(bandwidth_t.item())

        B = int(D.shape[0])
        N = int(D.shape[1])
        M = int(D.shape[2])

        # determine device index
        if isinstance(dev, torch.device):
            dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
        else:
            dev_idx = torch.cuda.current_device()

        try:
            cuda.select_device(dev_idx)
        except Exception:
            pass

        try:
            props = torch.cuda.get_device_properties(dev_idx)
            max_threads = props.max_threads_per_block
        except Exception:
            max_threads = 1024

        SAFE_THREAD_CAP = min(max_threads, 256)
        threads_per_block = int(min(max(N, M), SAFE_THREAD_CAP)) if SAFE_THREAD_CAP > 0 else 1
        if threads_per_block <= 0:
            threads_per_block = 1

        blocks_per_grid = B
        n_passes = N + M - 1

        # Prepare padded D_, R_local, E buffers for backward kernel
        D_ = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D.to(torch.float32)

        R_local = R.to(torch.float32).clone()
        # set borders as original code expected
        R_local[:, :, -1] = -math.inf
        R_local[:, -1, :] = -math.inf
        R_local[:, -1, -1] = R_local[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=torch.float32, device=dev)
        E[:, -1, -1] = 1.0

        # We'll run backward in reversed-chunks: reversed anti-diagonal indices form [0..n_passes-1] reversed.
        # To avoid negative arithmetic on device, precompute reversed start indices on host and launch chunks accordingly.
        CHUNK_SIZE = 1024
        if max(N, M) > 512:
            CHUNK_SIZE = 512

        # create a list of reversed chunk starts (for reversed order)
        rev_starts = []
        # We'll iterate from last diagonal (n_passes-1) down to 0 in chunks:
        p = n_passes
        while p > 0:
            chunk = min(CHUNK_SIZE, p)
            p -= chunk
            rev_starts.append((p, chunk))  # p is start index of this chunk (in normal order), chunk is size

        # Launch backward chunks in the same reversed order computed above.
        try:
            for (start_idx, chunk_len) in rev_starts:
                # For backward kernel we need to feed reversed pass index base that backward kernel interprets the same way
                # We'll give it the (start_pass_reversed) computed as (n_passes - (start_idx + chunk_len))
                start_pass_rev = n_passes - (start_idx + chunk_len)
                cuda.select_device(dev_idx)
                compute_softdtw_backward_cuda[blocks_per_grid, threads_per_block](
                    cuda.as_cuda_array(D_),
                    cuda.as_cuda_array(R_local),
                    1.0 / gamma_val,
                    float(bandwidth_val),
                    N, M,
                    start_pass_rev, chunk_len,
                    cuda.as_cuda_array(E)
                )
        except Exception as e:
            print("[SoftDTW] CUDA backward failed with exception:", repr(e))
            print("[SoftDTW] Falling back to CPU implementation for backward.")
            # fallback to CPU path
            E_cpu = compute_softdtw_backward(D_.cpu().numpy(), R_local.cpu().numpy(), gamma_val, bandwidth_val)
            E = torch.tensor(E_cpu, device=dev).type(torch.float32)
            E_out = E[:, 1:N + 1, 1:M + 1]
            if dtype != torch.float32:
                E_out = E_out.to(dtype)
            grad_input = grad_output.view(-1, 1, 1).expand_as(E_out) * E_out
            return grad_input, None, None

        E_out = E[:, 1:N + 1, 1:M + 1]
        if dtype != torch.float32:
            E_out = E_out.to(dtype)
        grad_input = grad_output.view(-1, 1, 1).expand_as(E_out) * E_out
        return grad_input, None, None

# ----------------------------------------------------------------------------------------------------------------------
# CPU implementations (numba-optimized) - unchanged aside from minor safety
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0.0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < np.abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = - gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1.0
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                if 0 < bandwidth < np.abs(i - j):
                    continue
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = math.exp(a0)
                b = math.exp(b0)
                c = math.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


# ----------------------------------------------------------------------------------------------------------------------
# PyTorch wrappers and convenience interfaces
# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU autograd Function that wraps the numba implementation.
    """
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma_val = float(gamma)
        bandwidth_val = float(bandwidth)
        D_np = D.detach().cpu().float().numpy()
        R_np = compute_softdtw(D_np, gamma_val, bandwidth_val)
        R = torch.tensor(R_np, device=dev).type(dtype)
        ctx.save_for_backward(D, R, torch.tensor(gamma_val, device=dev), torch.tensor(bandwidth_val, device=dev))
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        D, R, gamma_t, bandwidth_t = ctx.saved_tensors
        gamma_val = float(gamma_t.item())
        bandwidth_val = float(bandwidth_t.item())
        D_np = D.detach().cpu().float().numpy()
        R_np = R.detach().cpu().float().numpy()
        E_np = compute_softdtw_backward(D_np, R_np, gamma_val, bandwidth_val)
        E = torch.tensor(E_np, device=grad_output.device).type(grad_output.dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


class SoftDTW(torch.nn.Module):
    """
    SoftDTW module with both CUDA and CPU implementations and feature parity with original file.
    """

    def __init__(self, use_cuda: bool, gamma=1.0, normalize=False, bandwidth=None, dist_func=None, alignment_postprocess: str = 'row'):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        if alignment_postprocess not in ('row', 'none'):
            raise ValueError("alignment_postprocess must be 'row' or 'none'")
        self.alignment_postprocess = alignment_postprocess

        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        assert bx == by
        assert dx == dy

        use_cuda = self.use_cuda

        # Check device limit: many GPUs limit block size to 1024
        if use_cuda and (lx > 4096 or ly > 4096):
            # extremely large sequences — force CPU
            print("SoftDTW: Cannot use CUDA because the sequence length is extremely large (>4096). Falling back to CPU.")
            use_cuda = False

        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x_exp = x.unsqueeze(2).expand(-1, n, m, d)
        y_exp = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x_exp - y_exp, 2).sum(3)

    def forward_with_cost_matrix(self, C, return_alignment: bool = False):
        assert C.dim() == 3, "Cost matrix C must be 3-dimensional (batch, N, M)"

        max_len = max(C.shape[1], C.shape[2])
        use_cuda = self.use_cuda
        if use_cuda and max_len > 4096:
            print(f"SoftDTW: Cannot use CUDA because a sequence length ({max_len}) is extremely large")
            use_cuda = False

        func_dtw = _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

        if self.normalize:
            raise ValueError("Normalization is not supported when providing a pre-computed cost matrix.")

        if not return_alignment:
            return func_dtw(C, self.gamma, self.bandwidth)

        C_req = C.clone().detach().requires_grad_(True)
        sdtw_vals = func_dtw(C_req, self.gamma, self.bandwidth)
        grad_outputs = torch.ones_like(sdtw_vals, device=sdtw_vals.device)
        A = torch.autograd.grad(sdtw_vals, C_req, grad_outputs=grad_outputs, retain_graph=True)[0]

        if self.alignment_postprocess == 'row':
            eps = 1e-9
            A = A.clamp_min(0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + eps)
        return sdtw_vals, A

    def forward(self, X, Y, return_alignment: bool = False):
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            if return_alignment:
                raise NotImplementedError("Alignment return is not implemented for normalize=True.")
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)

        if not return_alignment:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

        D_xy = self.dist_func(X, Y)
        D_req = D_xy.clone().detach().requires_grad_(True)
        sdtw_vals = func_dtw(D_req, self.gamma, self.bandwidth)
        grad_outputs = torch.ones_like(sdtw_vals, device=sdtw_vals.device)
        A = torch.autograd.grad(sdtw_vals, D_req, grad_outputs=grad_outputs, retain_graph=True)[0]

        if self.alignment_postprocess == 'row':
            eps = 1e-9
            A = A.clamp_min(0.0)
            A = A / (A.sum(dim=-1, keepdim=True) + eps)
        return sdtw_vals, A

# ----------------------------------------------------------------------------------------------------------------------
# Profiling helpers (unchanged except for reduced defaults)
# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    from timeit import default_timer as timer
    start = timer()
    forward = sdtw(a, b)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()
    t += end - start
    return t, forward, grads


def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    # default to GPU if available, but tests are smaller to avoid timeouts
    sdtw = SoftDTW(torch.cuda.is_available(), gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 4

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda() if torch.cuda.is_available() else a_cpu
        b_gpu = b_cpu.cuda() if torch.cuda.is_available() else b_cpu

        # GPU
        if torch.cuda.is_available():
            t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)
        else:
            t_gpu, forward_gpu, backward_gpu = None, None, None

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # If GPU available, verify results (tolerances relaxed for larger sizes)
        if torch.cuda.is_available():
            assert torch.allclose(forward_cpu, forward_gpu.cpu(), atol=1e-5), "forward mismatch CPU vs GPU"
            assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward), "backward mismatch CPU vs GPU"

        if i > 0:
            times_cpu.append(t_cpu)
            if t_gpu is not None:
                times_gpu.append(t_gpu)

    avg_cpu = np.mean(times_cpu) if len(times_cpu) else float('nan')
    avg_gpu = np.mean(times_gpu) if len(times_gpu) else float('nan')
    print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    if not np.isnan(avg_cpu) and not np.isnan(avg_gpu) and avg_gpu > 0:
        print("  Speedup: ", avg_cpu / avg_gpu)
    print()


if __name__ == "__main__":
    from timeit import default_timer as timer
    torch.manual_seed(1234)

    # NOTE: use SMALLER profiles by default to avoid driver timeouts in CI / cluster
    profile(64, 17, 15, 2, tol_backward=1e-6)
    profile(128, 64, 64, 2, tol_backward=1e-4)
    profile(128, 256, 256, 2, tol_backward=1e-3)
