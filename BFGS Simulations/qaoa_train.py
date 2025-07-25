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
import qaoa_opt as qopt

def flatten_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_sols(diags, N):
    """
    Computes from solutions given the Hamiltonian
    """
    m = np.min(diags)
    res = []
    for i in range(len(diags)):
        if diags[i] == m:
            res.append(i)
    sols = []
    for i in res:
        sols.append(np.binary_repr(i, N))
    return sols

def get_metrics(x, terms, N):

    simclass = qokit.fur.choose_simulator(name='auto')
    sim = simclass(N, terms=terms, backend_options={'mpi': True})

    diags = sim.get_cost_diagonal()
    ground_truth = np.min(diags)
    bin_sols = get_sols(diags, N)
    dec_sols = [int(sol, 2) for sol in bin_sols]

    _result = sim.simulate_qaoa(x[:p], x[p:])
    probs = sim.get_probabilities(_result)

    prob_sum = 0
    for i in dec_sols:
        prob_sum += probs[i]

    return prob_sum, ground_truth, diags

def schatten_1norm_diagonal(diag): # Computes Schatten 1-norm for diagonal matrices
    sum = 0
    for i in range(len(diag)):
        cur = np.abs(diag[i])
        sum += cur
    return sum

# Mapping from optimizer names to functions and their accepted arguments
OPTIMIZER_MAP = {
    'grad_descent': (qopt.grad_descent, [
        'lr', 'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift', 'mode',
        'callback', 'track_success', 'verbose', 'f_calls', 'track_calls'
    ]),
    'layerwise_descent': (qopt.layerwise_descent, [
        'lr', 'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift', 'layers_per_step', 'mode',
        'callback', 'track_success', 'verbose', 'f_calls', 'track_calls'
    ]),
    'adam': (qopt.adam_opt, [
        'lr', 'beta1', 'beta2', 'eps', 'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift',
        'mode', 'callback', 'track_success', 'verbose', 'f_calls', 'track_calls'
    ]),
    'layerwise_adam': (qopt.layerwise_adam, [
        'lr', 'beta1', 'beta2', 'eps', 'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift',
        'mode', 'callback', 'layers_per_step', 'track_success', 'verbose', 'f_calls', 'track_calls'
    ]),
    'cqng': (qopt.cqng_descent, [
        'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift', 'reg', 'eps', 'callback', 'verbose', 'f_calls', 'track_calls', 'track_success'
    ]),
    'bayesian_layerwise_adam': (qopt.bayesian_layerwise_adam, [
        'lr', 'max_iters', 'tol', 'tol_patience', 'grad_fn', 'shift', 'mode', 'callback',
        'track_success', 'verbose', 'f_calls', 'track_calls', 'beta1', 'beta2', 'eps',
        'layer_prob_decay', 'num_layers_to_update'
    ]),
    'adaptive_exploration_layerwise_adam': (qopt.adaptive_exploration_layerwise_adam, [
        'lr', 'max_iters', 'tol', 'tol_patience', 'uncertainty_threshold', 'grad_fn', 'shift', 'mode',
        'callback', 'track_success', 'verbose', 'f_calls', 'track_calls', 'beta1', 'beta2', 'eps',
        'num_layers_to_update', 'min_exploration', 'max_exploration',
        'exploration_decay', 'exploration_growth', 'patience'
    ]),
    'bfgs_qaoa': (qopt.bfgs_qaoa, ['grad_fn', 'max_iters', 'callback', 'track_success', 'track_success'])
}

def train_QAOA(f, N, p, method='adam', hyperparams=None):
    """
    Trains the QAOA algorithm
    
    Parameters:
        f: objective function
        N: number of qubits (for compatibility / logging)
        p: QAOA depth
        method: string name of optimizer
        hyperparams: optional dictionary of hyperparameters

    Returns:
        tuple: (nit, success, final_value, parameters, total_time)
    """
    if hyperparams is None:
        initial_gamma = -1 * np.linspace(0, 1, p)
        initial_beta = np.linspace(1, 0, p)
        x0 = np.hstack([initial_gamma, initial_beta])
        hyperparams = {'x0': x0}

    elif 'x0' not in hyperparams:
        initial_gamma = -1 * np.linspace(0, 1, p)
        initial_beta = np.linspace(1, 0, p)
        x0 = np.hstack([initial_gamma, initial_beta])
        
    else: 
        x0 = hyperparams.get('x0')
        

    if method in OPTIMIZER_MAP:
        optimizer_fn, allowed_keys = OPTIMIZER_MAP[method]
        
        # Filter hyperparameters to pass only valid ones
        optimizer_kwargs = {
            k: v for k, v in hyperparams.items() if k in allowed_keys
        }
        res = optimizer_fn(f, x0, **optimizer_kwargs)
        
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return res