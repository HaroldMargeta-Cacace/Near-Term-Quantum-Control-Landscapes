import numpy as np
import qokit
import pandas as pd
import pickle
from tqdm import tqdm
from qokit import get_qaoa_objective

def grad(f, x, cur_calls=0, track_calls=True):
    cur_calls += 2 * len(x)
    return nd.Gradient(f)(x), cur_calls

file = "toy_data"
with open(file + ".pkl", 'rb') as f:
    df = pickle.load(f)

n_samples = 100

# Preallocate output arrays
loss_var = np.zeros(len(df))
grad_var = np.zeros(len(df))
grad_var_components = []

loop = tqdm(range(len(df)), desc="Starting...", position=0)

for i in loop:
    N = int(df["N"][i])
    terms = df["terms"][i]
    p = int(df["p"][i])
    f = get_qaoa_objective(N, terms=terms, parameterization='theta')

    loop.set_description(f"Collecting Gradients for {i}th Problem")

    # Preallocate for samples
    loss_evaluations = np.zeros(n_samples)
    grad_evaluations = np.zeros((n_samples, 2 * p))

    for j in range(n_samples):
        param = 2 * np.pi * np.random.rand(2 * p)

        loss_evaluations[j] = f(param)
        grad_eval, _ = grad(f, param, track_calls=False)
        grad_evaluations[j] = grad_eval

    loss_var[i] = np.var(loss_evaluations, ddof=1) # loss variance
    components = np.var(grad_evaluations, axis=0, ddof=1) # component variance
    grad_var_components.append(components)
    grad_var[i] = np.mean(components) # mean component variance

df['sample_loss_var'] = loss_var
df['sample_grad_var'] = grad_var
df['sample_gar_var_components'] = grad_var_components

df.to_csv(file + '_wvars.csv')
df.to_pickle(file + '_wvars.pkl')


