import taichi as ti
import numpy as np
import torch
import time

# ------------------ Hyperparameter Setting ------------------
B = 4   # batch size
n = 32   # input dimension D
m = 8   # output dimension D'
device = 'cpu'

# ------------------ Device Initialization ------------------
if device == 'gpu':
    ti.init(arch=ti.gpu)
    torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
else:
    ti.init(arch=ti.cpu)
    torch_device = torch.device("cpu")

# ------------------ Taichi Kernels ------------------
@ti.kernel
def compute_y_batch():
    # For each batch we compute the result
    for b_ in ti.ndrange(B):
        y[b_] = ti.tanh(W[None] @ x[b_] + bias[None])

@ti.kernel
def clear_gradients():
    # Reset Gradient Wait I'm confused here
    for b_ in ti.ndrange(B):
        x.grad[b_].fill(0.0)
        y.grad[b_].fill(0.0)

@ti.kernel
def set_v_grad():
    for b_ in ti.ndrange(B):
        y.grad[b_] = v[b_]

@ti.kernel
def copy_grad_to_result():
    for b_ in ti.ndrange(B):
        JVP_result[b_] = x.grad[b_]

def taichi_compute_jvp():
    clear_gradients()
    set_v_grad()
    compute_y_batch()
    compute_y_batch.grad()
    copy_grad_to_result()
    return JVP_result.to_numpy()

# ------------------ PyTorch Implementation (direct J calculation) ------------------
def pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch):
    def forward_fn(x_):
        return torch.tanh(x_ @ W_torch.T + b_torch)
    
    jac = torch.autograd.functional.jacobian(forward_fn, x_torch) #(B,D',B,D)

    #Check, a different implementation of torch JVP before
    #jac = jac.transpose(1, 2).diagonal(dim1=0, dim2=1).movedim(-1, 0)
    #return torch.einsum('bmn,bm->bn', jac, v_torch).cpu().numpy()

    #This is following Previous Procedure, I think previously sth is wrong (double check)
    J = torch.sum(jac, dim=2) #(B,D',D)
    #v_torch: (B,D') -> (B,1,D')
    return torch.einsum('bij, bjk -> bik', v_torch.unsqueeze(1), J).squeeze(1).cpu().numpy() # (B,D)

if __name__ == "__main__":
    torch.manual_seed(42)

    # Taichi Object Initialization (or we should say object container)
    W = ti.Matrix.field(m, n, dtype=ti.f32, shape=())  
    bias = ti.Vector.field(m, dtype=ti.f32, shape=())  
    # No shape for W and b? cause regardless of batch, the model is the same
    x = ti.Vector.field(n, dtype=ti.f32, shape=(B,), needs_grad=True) # Input (B,D)
    y = ti.Vector.field(m, dtype=ti.f32, shape=(B,), needs_grad=True) # Output (B,D')
    v = ti.Vector.field(m, dtype=ti.f32, shape=(B,)) # The vector (B,D') for JVP
    JVP_result = ti.Vector.field(n, dtype=ti.f32, shape=(B,))
    # J^T: (B,D,D'), v: (B,D') -> JVP_res: (B,D)

    # Taichi 
    W_np = torch.randn(m, n, dtype=torch.float32).numpy()
    bias_np = torch.randn(m, dtype=torch.float32).numpy()
    x_np = torch.randn(B, n, dtype=torch.float32).numpy()
    v_np = torch.randn(B, m, dtype=torch.float32).numpy()

    W[None] = W_np
    bias[None] = bias_np
    for b_ in range(B):
        x[b_] = x_np[b_]
        v[b_] = v_np[b_]
    
    _ = taichi_compute_jvp()

    x_torch = torch.tensor(x_np, dtype=torch.float32, device=torch_device)

    # ------------------ Speed Testing ------------------
    taichi_start = time.time()
    taichi_res = taichi_compute_jvp()
    taichi_time = time.time() - taichi_start

    W_torch = torch.tensor(W_np, dtype=torch.float32, device=torch_device)
    b_torch = torch.tensor(bias_np, dtype=torch.float32, device=torch_device)
    v_torch = torch.tensor(v_np, dtype=torch.float32, device=torch_device)
    
    torch_start = time.time()
    torch_res = pytorch_compute_jvp(x_torch, W_torch, b_torch, v_torch)
    torch_time = time.time() - torch_start

    max_error = np.max(np.abs(taichi_res - torch_res))
    print("\n================ Results Verification ================")
    print(f"Maximum Absolute Error: {max_error:.2e}")
    print(f"Taichi JVP Result, Sample 0, First 5 Dimension: {taichi_res[0, :5]}")
    print(f"Pytorch JVP Result, Sample 0, First 5 Dimension: {torch_res[0, :5]}")

    print("\n================ Speed Comparison ================")
    print(f"Taichi Computation Time: {taichi_time * 1000:.2f} ms")
    print(f"PyTorch Computation Time: {torch_time * 1000:.2f} ms")
    print(f"Speed Ratio: {torch_time/taichi_time:.2f}x")