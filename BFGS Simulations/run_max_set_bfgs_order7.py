import numpy as np
import scipy
import networkx as nx
import pickle
import pandas as pd
from tqdm import tqdm
import time
import os

import numdifftools as nd
import qokit
from qokit import get_qaoa_objective
import qaoa_opt as qopt
import qaoa_train as qtrain
import qaoa_data_manager as qmngr
import qaoa_graph_features as gfeat

def get_mis_terms(G: nx.Graph):
    l = 1
    terms_a = [(-1, tuple([i])) for i in G.nodes()]
    terms_b = [(-G.number_of_nodes(), tuple())]
    terms_c = [(l, (int(e[0]), int(e[1]))) for e in G.edges()]
    terms_d = [(l, tuple([int(e[0])])) for e in G.edges()]
    terms_e = [(l, tuple([int(e[1])])) for e in G.edges()]
    terms_f = [(l * G.number_of_edges(), tuple())]
    return terms_a + terms_b + terms_c + terms_d + terms_e + terms_f


def get_sols(diag, N):
    """Return all bitstrings with minimum cost."""
    min_val = np.min(diag)
    return [format(i, f'0{N}b') for i, e in enumerate(diag) if e == min_val]

# Configure the runs
filelist = ['Dataset_Up_To_Order_7_a','Dataset_Up_To_Order_7_b',
            'Dataset_Up_To_Order_7_c']
all_dicts = qmngr.load_and_sort_graphs(filelist)
p_list = [10, 15]
optimizer_names = ['bfgs_qaoa']
graph_dicts, untested_dicts = qmngr.filter_untested_graphs(all_dicts, p_list, optimizer_names)
df = pd.DataFrame()
graph_loop = tqdm(range(len(graph_dicts)), desc="Starting...", position=0)
outnames = qmngr.generate_output_filenames(graph_dicts, p_list, optimizer_names, filelist)

# Elements of the dictionary outnames
# 'base_name': f"{base_name}",
# 'csv': f"{base_name}.csv",
# 'pkl': f"{base_name}.pkl",
# 'summary_csv': f"{base_name}_summary.csv",
# 'summary_pkl': f"{base_name}_summary.pkl",
# 'complete_csv': f"{base_name}_complete.csv",
# 'complete_pkl':  f"{base_name}_complete.pkl",
# 'feature_correlations_csv': f"{base_name}_feature_exp_correlations.csv",
# 'feature_correlations_pkl': f"{base_name}_feature_exp_correlations.pkl",
# 'graphs': "_".join(input_filenames) + "_features"

for i in graph_loop:
    G = nx.Graph(graph_dicts[i])
    terms = get_mis_terms(G)
    adj = nx.to_numpy_array(G)

    N = G.number_of_nodes()
    simclass = qokit.fur.choose_simulator(name='auto')
    sim = simclass(N, terms=terms)

    diags = sim.get_cost_diagonal()
    ground_truth = np.min(diags)
    hamil_schatt_1 = qtrain.schatten_1norm_diagonal(diags)

    for p in p_list:
        # optimizer_names = ['bfgs_qaoa'] Here as a reminder, but unnecessary
        # Add or remove hyperparameters as needed
        optimizer_hyperparams = {
            'grad_descent': {'lr': 0.001},
            'layerwise_descent': {'lr': 0.001, 'layers_per_step': max(1, p // 2)},
            'adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
            'layerwise_adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'layers_per_step': max(1, p // 2)},
            'cqng': {'lr': 0.001},
            'bayesian_layerwise_adam': {'lr': 0.001, 'num_layers_to_update': max(1, 3 * p // 4)},
            'adaptive_exploration_layerwise_adam': {'lr': 0.001},
            'bfgs_qaoa': {'max_iters': 1000}
        }
        for opt_name in optimizer_names:
            if p in untested_dicts[i] and opt_name in untested_dicts[i].get(p):
                f = get_qaoa_objective(N, terms=terms, parameterization='theta')

                graph_loop.set_description(f"Training QAOA (p={p}, opt={opt_name})")
                hyperparams = optimizer_hyperparams[opt_name].copy()
                hyperparams['grad_fn'] = lambda x, c, l=None: qopt.grad(f, x, cur_calls=c, grad_list=l)

                start = time.time()
                try:
                    out, history, nit, calls, success = qtrain.train_QAOA(
                        f, N, p, method=opt_name, hyperparams=hyperparams
                    )
                except Exception as e:
                    print(f"Error on graph with N={N}, p={p}, optimizer={opt_name}: {e}")
                    continue
                runtime = time.time() - start

                x = history[-1][0]
                exp = f(x)
                approx = exp / ground_truth
                p_error = np.abs(exp - ground_truth) / np.abs(ground_truth) * 100

                bin_sols = get_sols(diags, N)
                dec_sols = [int(sol, 2) for sol in bin_sols]
                f_calls_depth = p * calls
                result = sim.simulate_qaoa(x[:p], x[p:])
                probs = sim.get_probabilities(result)
                prob_sum = sum(probs[i] for i in dec_sols)

                param_list, grad_list = zip(*history)
                # Stack into (T x D) array: each row is a gradient vector at one step
                grad_matrix = np.vstack(grad_list)

                # Compute variance across steps for each parameter
                grad_variances = np.var(grad_matrix, axis=0)
                mean_grad_variance = np.mean(grad_variances)
                new_row = pd.DataFrame({
                    'N': [N],
                    'adj': [adj],
                    'terms': [terms],
                    'p': [p],
                    'optimizer': [opt_name],
                    'mixer': ['x'],
                    'hamiltonian_schatt_1_norm': [hamil_schatt_1],
                    'nit': [nit],
                    'f_calls': [calls],
                    'f_calls_depth': [f_calls_depth],
                    'runtime': [runtime],
                    'success': [int(success)],
                    'parameter_list': [param_list],
                    'grad_list': [grad_list],
                    'mean_grad_variance': [mean_grad_variance],
                    'grad_variances': [grad_variances],
                    'sigma_p': [prob_sum],
                    'approx': [approx],
                    'percent_error': [p_error]
                })
                
                df = pd.concat([df, new_row], ignore_index=True)
                
    # Create temporary saves every 25 iterations
    itersiz = 25
    if (i + 1) % itersiz == 0:
        qmngr.save_results_dataframe(df, outnames, f'temp{i + 1}')

        if (i + 1) / 25 > 1:
            try:
                os.remove(outnames['base_name'] + f'temp{i + 1 - itersiz}.csv')
            except FileNotFoundError:
                print(f"Temp CSV file temp{i + 1 - itersiz}.csv not found for removal.")
            except Exception as e:
                print(f"An error occurred while removing temp{i + 1 - itersiz}.csv: {e}")

            try:
                os.remove(outnames['base_name'] + f'temp{i + 1 - itersiz}.pkl')
            except FileNotFoundError:
                print(f"Temp PKL file temp{i + 1 - itersiz}.pkl not found for removal.")
            except Exception as e:
                print(f"An error occurred while removing temp{i}.pkl: {e}")
                
# SAVE
qmngr.save_results_dataframe(df, outnames)
qmngr.save_summary_statistics(df, outnames)
qmngr.save_complete_results(df, outnames)

# Remove temp saves
try:
    os.remove(outnames['base_name'] + f'temp{len(graph_dicts)}.csv')
except FileNotFoundError:
    print("Final temp CSV file not found for removal.")
except Exception as e:
    print(f"An error occurred during final CSV cleanup: {e}")

try:
    os.remove(outnames['base_name'] + f'temp{len(graph_dicts)}.pkl')
except FileNotFoundError:
    print("Final temp PKL file not found for removal.")
except Exception as e:
    print(f"An error occurred during final PKL cleanup: {e}")
