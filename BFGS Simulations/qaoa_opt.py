import numpy as np
import scipy
import networkx as nx
import numdifftools as nd
import pickle
import pandas as pd
from tqdm import tqdm
import time

import qokit
from qokit import get_qaoa_objective

# Parameter-shift method currently nonfunctional. Use grad instead.
def parameter_shift_gradient(f, x, shift=np.pi/2, cur_calls=0, 
                            track_calls=True, grad_list=None):
    gradient = np.zeros_like(x)
    
    if grad_list is None:
        grad_list = [j for j in range(len(x))]
        
    for i in grad_list:
        shift_vector = np.zeros_like(x)
        shift_vector[i] = shift
        gradient[i] = 0.5 * (f(x + shift_vector) - f(x - shift_vector))
        
    if track_calls:
        cur_calls += 2*len(grad_list)
        return gradient, cur_calls
    else:
        return gradient

def grad(f, x, cur_calls=0, track_calls=True, method='central', grad_list=None): 
    if grad_list is None:
        if track_calls:
            cur_calls += (1 + int(method=='central')) * len(x) # Note: complex difference unsupported
            return nd.Gradient(f)(x), cur_calls
        
        else:
            return nd.Gradient(f, method=method)(x)
    
    else:
        grad = np.zeros_like(x)
        dx = 1e-6
        for i in grad_list: # Not efficient, but not made for many parameters
            x_plus = np.array(x)
            x_minus = np.array(x)
            x_plus[i] += dx
            x_minus[i] -= dx
            if method == 'central':
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * dx)
                cur_calls += 2
            elif method == 'forward':
                grad[i] = (f(x_plus) - f(x)) / dx
                cur_calls += 1
            elif method == 'backward':
                grad[i] = (f(x) - f(x_minus)) / dx
                cur_calls += 1
            else:
                raise ValueError("Invalid method. Choose 'central', 'forward', or 'backward'.")
        return grad, cur_calls if track_calls else grad


def grad_descent(
    f,                          # Objective function
    x0,                         # Initial parameters (1D numpy array)
    lr=0.01,                     # Learning rate
    max_iters=1000,             # Maximum number of iterations
    tol=1e-4,                  # Convergence tolerance
    tol_patience=3,
    grad_fn=None,              # Gradient function (optional)
    shift=np.pi/2,             # Shift for parameter shift rule
    mode='min',                # Whether to do gradient descent or ascent
    callback=None,             # Optional logging callback
    track_success=True,        # Output whether optimization converged
    verbose=False,               # Whether to show progress bar
    f_calls=0,                 # How many calls there are initially
    track_calls=True          # Whether to track circuit calls
):
    x = x0.copy()
    
    if mode=='min':
        sgn = -1
    else: 
        sgn = 1
    
    history = []
    success = False
    if grad_fn is None:
        grad_fn = lambda x, c: parameter_shift_gradient(f, x, shift, c)

    iterator = range(max_iters)
    nit = max_iters
    
    if verbose:
        iterator = tqdm(iterator, desc="Gradient Descent")
        
    pat_count = 1
    was_under = True
    for i in iterator:
        grad, f_calls = grad_fn(x, f_calls)
        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(i, x, grad)

        if np.linalg.norm(grad) < tol:
            if was_under:
                pat_count += 1
            else:
                pat_count = 1
                
            if pat_count >= tol_patience:
                success = True
                nit = i
                break
            
            was_under = True
        
        else:
            was_under = False
        
        x = x + sgn * lr * grad

    if track_calls:
        if track_success:
            return x, history, nit, f_calls, success
        return x, history, nit, f_calls
    
    else: 
        if track_success:
            return x, history, nit, success
        return x, history, nit
    
def layerwise_descent( 
    f,                          # Objective function
    x0,                         # Initial parameters: np.hstack([gamma, beta])
    lr=0.01,                     # Learning rate
    max_iters=1000,             # Maximum number of iterations
    tol=1e-4,                   # Convergence tolerance (on gradient norm)
    tol_patience=3,
    grad_fn=None,              # Gradient function: grad(x, f_calls) -> (grad, f_calls)
    shift=np.pi/2,             # Shift for parameter shift
    layers_per_step=1,         # How many layers to update per step
    mode='min',                # 'min' for gradient descent, 'max' for ascent
    callback=None,             # Optional logging callback (i, x, grad)
    track_success=True,        # Whether to return convergence status
    verbose=False,             # Show progress bar
    f_calls=0,                 # Initial function call count
    track_calls=True           # Whether to return f_calls
):
    x = x0.copy()
    p = len(x) // 2
    history = []
    success = False

    sgn = -1 if mode == 'min' else 1

    if grad_fn is None:
        grad_fn = lambda x, c, l: parameter_shift_gradient(f, x, shift, cur_calls=c, grad_list=l)

    iterator = range(max_iters)
    nit = max_iters
    if verbose:
        iterator = tqdm(iterator, desc="Layerwise Gradient Descent")
    pat_count = 1
    was_under = True
    for i in iterator:
        # Sample which QAOA layers to update
        sampled_layers = np.random.choice(p, size=layers_per_step, replace=False)
        
        grad, f_calls = grad_fn(x, f_calls)
        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(i, x, grad)

        if np.linalg.norm(grad) * p / (layers_per_step) < tol:
            if was_under:
                pat_count += 1
                
            else:
                pat_count = 1
                
            if pat_count >= tol_patience:
                success = True
                nit = i
                break
            
            was_under = True
            
        else:
            was_under = False
        
        for l in sampled_layers:
            x[l]     += sgn * lr * grad[l]       # gamma
            x[p + l] += sgn * lr * grad[p + l]   # beta

    if track_calls:
        return (x, history, nit, f_calls, success) if track_success else (x, history, nit, f_calls)
    else:
        return (x, history, nit, success) if track_success else (x, history, nit)
    
def adam_opt(
    f,                          # Objective function
    x0,                         # Initial parameters (1D numpy array)
    lr=0.01,                    # Learning rate (α)
    beta1=0.9,                  # Exponential decay for first moment
    beta2=0.999,                # Exponential decay for second moment
    eps=1e-8,                   # Small value to prevent division by zero
    max_iters=1000,             # Maximum number of iterations
    tol=1e-4,                   # Convergence tolerance on grad norm\
    tol_patience=3,
    grad_fn=None,              # Gradient function: grad(x, f_calls) -> (grad, f_calls)
    shift=np.pi/2,             # Parameter shift value
    mode='min',                # 'min' for descent, 'max' for ascent
    callback=None,             # Callback(i, x, grad)
    track_success=True,        # Return success status
    verbose=False,             # Show tqdm progress
    f_calls=0,                 # Initial function call count
    track_calls=True           # Whether to track circuit calls
):
    x = x0.copy()
    m = np.zeros_like(x)   # First moment vector
    v = np.zeros_like(x)   # Second moment vector
    history = []
    success = False

    sgn = -1 if mode == 'min' else 1

    if grad_fn is None:
        grad_fn = lambda x, c: parameter_shift_gradient(f, x, shift, c)

    iterator = range(max_iters)
    nit = max_iters
    
    if verbose:
        iterator = tqdm(iterator, desc="ADAM Optimization")
    
    pat_count = 1
    was_under = True
    for t in iterator:
        grad, f_calls = grad_fn(x, f_calls)
        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(t, x, grad)

        if np.linalg.norm(grad) < tol:
            if was_under:
                pat_count += 1
            
            else:
                pat_count = 1
                
            if pat_count >= tol_patience:
                success = True
                nit = t
                break
            
            was_under = True
        
        else: 
            was_under = False

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        root_v_hat = np.sqrt(v_hat)
        
        # Update parameters
        x = x + sgn * lr * m_hat / (np.sqrt(v_hat) + eps)

    if track_calls:
        return (x, history, nit, f_calls, success) if track_success else (x, history, nit, f_calls)
    else:
        return (x, history, nit, success) if track_success else (x, history, nit)
    
def layerwise_adam(
    f,                          # Objective function
    x0,                         # Initial parameters (gamma + beta)
    lr=0.01,                    # Learning rate
    beta1=0.9,                  # ADAM first moment decay
    beta2=0.999,                # ADAM second moment decay
    eps=1e-8,                   # Epsilon to avoid divide by zero
    max_iters=1000,             # Max optimization iterations
    tol=1e-4,                   # Gradient norm threshold for convergence
    tol_patience=3,
    grad_fn=None,              # Gradient function: grad(x, f_calls) -> grad, f_calls
    shift=np.pi/2,             # Parameter shift rule shift
    mode='min',                # 'min' or 'max'
    callback=None,             # Optional callback(i, x, grad)
    layers_per_step=1,         # Number of (gamma, beta) layer-pairs to update per step
    track_success=True,        # Whether to return success flag
    verbose=False,             # Whether to show tqdm bar
    f_calls=0,                 # Initial function call count
    track_calls=True           # Track total function calls
):
    x = x0.copy()
    p = len(x) // 2
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    success = False
    sgn = -1 if mode == 'min' else 1

    if grad_fn is None:
        grad_fn = lambda x, c, l: parameter_shift_gradient(f, x, shift, c, grad_list=l)

    layer_indices = list(range(p))
    iterator = range(max_iters)
    nit = max_iters
    if verbose:
        iterator = tqdm(iterator, desc="Layerwise ADAM")
    pat_count = 1
    was_under = True
    for t in iterator:
        # Sample layers to update
        sampled_layers = np.random.choice(layer_indices, size=layers_per_step, replace=False)
        
        param_indices = [i for l in sampled_layers for i in [l, l + p]]
        grad, f_calls = grad_fn(x, f_calls, param_indices)
        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(t, x, grad)

        if np.linalg.norm(grad) * p / (layers_per_step) < tol:
            if was_under:
                pat_count += 1
            else: 
                pat_count = 1
                
            if pat_count >= tol_patience:
                success = True
                nit = t
                break
            
            was_under = True
            
        else: 
            was_under = False

        for l in sampled_layers:
            # gamma index: l, beta index: l + p
            for i in [l, l + p]:
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * (grad[i] ** 2)

                m_hat = m[i] / (1 - beta1 ** (t + 1))
                v_hat = v[i] / (1 - beta2 ** (t + 1))

                x[i] += sgn * lr * m_hat / (np.sqrt(v_hat) + eps)

    if track_calls:
        return (x, history, nit, f_calls, success) if track_success else (x, history, nit, f_calls)
    else:
        return (x, history, nit, success) if track_success else (x, history, nit)

    
def cqng_descent(
    f,                          # Objective function
    x0,                         # Initial parameters (1D numpy array)
    max_iters=1000,             # Max iterations
    tol=1e-4,                  # Gradient norm tolerance
    tol_patience=2,
    grad_fn=None,              # Gradient function
    shift=np.pi/2,             # Parameter-shift rule shift
    reg=1e-3,                  # Regularization for F_inv (QFIM)
    eps=1e-8,                  # Prevent division by zero + improve numerical stability
    callback=None,             # Optional logging callback
    verbose=False,             # Whether to show progress bar
    f_calls=0,                 # Initial function call count
    track_calls=True,          # Whether to track circuit calls
    track_success=True         # Whether to track convergence status
):
    x = x0.copy()
    history = []
    success = False
    dt_prev = None
    ng_prev = None

    def line_search(f, x, dt, f_calls=0, track_calls=True):
        def obj(alpha):
            nonlocal f_calls
            if track_calls:
                f_calls += 1
            return f(x + alpha * dt)

        res = scipy.optimize.minimize_scalar(obj, bracket=[0, 1])
        return res.x, res.nfev, f_calls

    iterator = range(max_iters)
    nit = max_iters
    if verbose:
        iterator = tqdm(iterator, desc="CQNG Descent")
        
    if grad_fn is None:
        grad_fn = lambda x, c: parameter_shift_gradient(f, x, shift, c)

    pat_count = 1
    was_under = True
    for i in iterator:
        # Gradient via parameter shift
        grad, f_calls = grad_fn(x, f_calls)

        if np.linalg.norm(grad) < tol:
            if was_under:
                pat_count += 1
                
            else: 
                pat_count = 1
            
            if pat_count >= tol_patience:
                success = True
                nit = i
                break
            
            was_under = True
        
        else:
            was_under = False

        # Diagonal approximation to QFIM
        F_diag = grad**2 + reg
        F_inv = 1.0 / F_diag
        ng = F_inv * grad

        # Conjugate direction
        if dt_prev is None:
            dt = -ng
        else:
            y = ng - ng_prev
            beta = max(0, np.dot(ng, y) / (np.dot(ng_prev, ng_prev) + eps))
            dt = -ng + beta * dt_prev

        # Line search to determine step size
        alpha, line_calls = line_search(f, x, dt, f_calls)
        f_calls += line_calls

        # Update parameters
        x = x + alpha * dt

        # Save iteration info
        history.append((x.copy(), grad.copy()))
        dt_prev = dt.copy()
        ng_prev = ng.copy()

        if callback is not None:
            callback(i, x, grad)

    if track_calls:
        if track_success:
            return x, history, nit, f_calls, success
        return x, history, nit, f_calls
    else:
        if track_success:
            return x, history, nit, success
        return x, history, nit
    
def bayesian_layerwise_adam(
    f, x0, lr=0.01, max_iters=1000, tol=1e-4, tol_patience=2,
    grad_fn=None, shift=np.pi/2, mode='min',
    callback=None, track_success=True, verbose=False,
    f_calls=0, track_calls=True, beta1=0.9, beta2=0.999, eps=1e-8,
    layer_prob_decay=0.9, num_layers_to_update=None
):
    x = x0.copy()
    p = len(x0) // 2
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    nit = max_iters

    grad_magnitude_ema = np.zeros(p)
    grad_variance_ema = np.ones(p)  # initialize to avoid 0s
    history = []
    success = False
    rng = np.random.default_rng()

    if grad_fn is None:
        def grad_fn(x, cur_calls, grad_list=None):
            return parameter_shift_gradient(f, x, shift=shift, cur_calls=cur_calls,
                                            track_calls=track_calls, grad_list=grad_list)


    tol_hit_count = 0
    first_iter = True

    for i in tqdm(range(max_iters), disable=not verbose):
        t += 1

        if first_iter:
            # First iteration: full gradient on all parameters
            grad_indices = list(range(len(x)))
            grad, f_calls = grad_fn(x, f_calls, grad_indices)

            # Initialize EMAs from full gradient per layer
            for idx in range(p):
                layer_grad = np.array([grad[idx], grad[p + idx]])
                grad_magnitude_ema[idx] = np.linalg.norm(layer_grad)
            grad_variance_ema = np.ones(p)

            top_k_layers = np.arange(p)
            first_iter = False
        else:
            # Bayesian acquisition scores
            acquisition_scores = grad_magnitude_ema * np.sqrt(grad_variance_ema + eps)

            if num_layers_to_update is None:
                k = p
            else:
                k = num_layers_to_update

            # Sample top-k layers by acquisition score
            top_k_layers = np.argsort(acquisition_scores)[-k:]
            grad_indices = np.concatenate([top_k_layers, top_k_layers + p])

            grad, f_calls = grad_fn(x, f_calls, grad_indices)

            # Update EMA stats and Adam moments per sampled layer
            for idx in top_k_layers:
                gamma_grad = grad[idx]
                beta_grad = grad[p + idx]
                layer_grad = np.array([gamma_grad, beta_grad])
                norm = np.linalg.norm(layer_grad)

                prev_mag = grad_magnitude_ema[idx]
                grad_magnitude_ema[idx] = (
                    layer_prob_decay * grad_magnitude_ema[idx] +
                    (1 - layer_prob_decay) * norm
                )
                grad_variance_ema[idx] = (
                    layer_prob_decay * grad_variance_ema[idx] +
                    (1 - layer_prob_decay) * (norm - prev_mag) ** 2
                )

                # Adam update for gamma and beta
                m[idx] = beta1 * m[idx] + (1 - beta1) * gamma_grad
                v[idx] = beta2 * v[idx] + (1 - beta2) * (gamma_grad ** 2)
                m[p + idx] = beta1 * m[p + idx] + (1 - beta1) * beta_grad
                v[p + idx] = beta2 * v[p + idx] + (1 - beta2) * (beta_grad ** 2)

        # Compute bias-corrected moments and step
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))

        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        sgn = -1 if mode == 'min' else 1
        x += sgn * step

        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(i, x, grad)

        # Compute avg layer grad norm for stopping check
        if first_iter:
            avg_layer_grad_norm = np.mean([np.linalg.norm([grad[j], grad[p + j]]) for j in range(p)])
        else:
            avg_layer_grad_norm = np.mean([np.linalg.norm([grad[j], grad[p + j]]) for j in top_k_layers])

        if avg_layer_grad_norm < tol:
            tol_hit_count += 1
            if tol_hit_count >= tol_patience:
                success = True
                nit = i
                break
        else:
            tol_hit_count = 0

    if track_calls:
        if track_success:
            return x, history, nit, f_calls, success
        return x, history, nit, f_calls
    else:
        if track_success:
            return x, history, nit, success
        return x, history, nit
    
def adaptive_exploration_layerwise_adam(
    f, x0, lr=0.01, max_iters=1000, tol=1e-4, tol_patience=2, uncertainty_threshold=0.1,
    grad_fn=None, shift=np.pi/2, mode='min',
    callback=None, track_success=True, verbose=False,
    f_calls=0, track_calls=True, beta1=0.9, beta2=0.999, eps=1e-8,
    num_layers_to_update=None,
    min_exploration=0.05, max_exploration=1,
    exploration_decay=0.95, exploration_growth=1.2, patience=5, init_prob=None
):
    x = x0.copy()
    p = len(x0) // 2
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    nit = max_iters

    grad_magnitude_ema = np.zeros(2 * p)
    grad_variance_ema = np.ones(2 * p)  # avoid zeros
    history = []
    success = False
    rng = np.random.default_rng()

    if grad_fn is None:
        def grad_fn(x, cur_calls, grad_list=None):
            return parameter_shift_gradient(f, x, shift=shift, cur_calls=cur_calls,
                                            track_calls=track_calls, grad_list=grad_list)

    # Initialize
    full_grad, f_calls = grad_fn(x, f_calls, list(range(len(x))))
    grad_magnitude_ema = np.abs(full_grad)
    grad_variance_ema = np.ones_like(x)

    exploration_rate = max_exploration
    no_improve_iters = 0
    last_fun_val = f(x)
    f_calls += 1

    pat_count = 1
    was_under = True

    for i in tqdm(range(max_iters), disable=not verbose):
        t += 1

        acquisition_scores = grad_magnitude_ema * np.sqrt(grad_variance_ema + eps)

        if num_layers_to_update is None:
            num_layers = p
        else:
            num_layers = min(num_layers_to_update, p)

        # Layer coupling: group (γ_l, β_l) as layer l
        layers = np.arange(p)
        num_explore = int(np.ceil(exploration_rate * num_layers))
        num_exploit = num_layers - num_explore

        exploit_layers = layers[np.argsort(
            np.add(acquisition_scores[:p], acquisition_scores[p:])
        )[-num_exploit:]] if num_exploit > 0 else np.array([], dtype=int)

        remaining_layers = np.setdiff1d(layers, exploit_layers)
        explore_layers = rng.choice(remaining_layers, size=num_explore, replace=False) if num_explore > 0 else np.array([], dtype=int)

        selected_layers = np.concatenate([exploit_layers, explore_layers])
        selected_indices = np.concatenate([selected_layers, selected_layers + p])  # (γ_l, β_l)

        grad, f_calls = grad_fn(x, f_calls, selected_indices)

        # Update EMAs
        grad_magnitude_ema[selected_indices] = (
            exploration_decay * grad_magnitude_ema[selected_indices] +
            (1 - exploration_decay) * np.abs(grad[selected_indices])
        )
        grad_variance_ema[selected_indices] = (
            exploration_decay * grad_variance_ema[selected_indices] +
            (1 - exploration_decay) * (grad[selected_indices] - grad_magnitude_ema[selected_indices]) ** 2
        )

        # ADAM update
        m[selected_indices] = beta1 * m[selected_indices] + (1 - beta1) * grad[selected_indices]
        v[selected_indices] = beta2 * v[selected_indices] + (1 - beta2) * (grad[selected_indices] ** 2)

        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))

        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        sgn = -1 if mode == 'min' else 1
        x += sgn * step

        history.append((x.copy(), grad.copy()))

        if callback is not None:
            callback(i, x, grad)

        fun_val = f(x)
        f_calls += 1

        # Adapt exploration rate
        improvement = last_fun_val - fun_val if mode == 'min' else fun_val - last_fun_val
        avg_uncertainty = np.mean(np.sqrt(grad_variance_ema))

        if improvement < tol or avg_uncertainty > uncertainty_threshold:
            no_improve_iters += 1
        else:
            no_improve_iters = 0

        if no_improve_iters >= patience:
            exploration_rate = min(max_exploration, exploration_rate * exploration_growth)
            no_improve_iters = 0
        else:
            exploration_rate = max(min_exploration, exploration_rate * exploration_decay)

        last_fun_val = fun_val

        # Patience-based stopping
        if np.linalg.norm(grad[selected_indices]) * p / len(selected_indices) < tol:
            if was_under:
                pat_count += 1
            else:
                pat_count = 1
            was_under = True
            if pat_count >= tol_patience:
                success = True
                nit = i
                break
        else:
            was_under = False

    if track_calls:
        if track_success:
            return x, history, nit, f_calls, success
        return x, history, nit, f_calls
    else:
        if track_success:
            return x, history, nit, success
        return x, history, nit
    
def bfgs_qaoa(
    f, x0, grad_fn=None, max_iters=1000,
    callback=None, track_calls=True, track_success=True
):
    f_calls = [0]  # mutable call counter
    history = []

    def wrapped_f(x):
        f_calls[0] += 1
        return f(x)

    if grad_fn is not None:
        def wrapped_grad(x):
            grad, new_calls = grad_fn(x, f_calls[0])
            f_calls[0] = new_calls
            history.append((x.copy(), grad.copy()))
            return grad
    else:
        def wrapped_grad(x):
            raise ValueError("Gradient function required for BFGS in this framework.")

    def wrapped_callback(xk):
        if callback is not None:
            callback(len(history), xk, None)

    result = scipy.optimize.minimize(
        wrapped_f,
        x0,
        method='BFGS',
        jac=wrapped_grad,
        options={'maxiter': max_iters},
        callback=wrapped_callback
    )

    x_opt = result.x
    nit = result.nit
    success = result.success

    if track_calls:
        if track_success:
            return x_opt, history, nit, f_calls[0], success
        return x_opt, history, nit, f_calls[0]
    else:
        if track_success:
            return x_opt, history, nit, success
        return x_opt, history, nit
