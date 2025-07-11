import numpy as np
import scipy
from numdifftools import Jacobian
import networkx as nx
import pickle
import pandas as pd
from tqdm import tqdm
import time

import qokit
from qokit import get_qaoa_objective

def load_graphs(filename):
   """
   Loads graphs from the pickle file
   Returns a list of dictionaries containing the adjacency matrix of each graph
   """
   with open(filename, 'rb') as f:
     graph_dicts = pickle.load(f)
   return graph_dicts

def get_mis_terms(G: nx.Graph):
  """
  Computes the cost Hamiltonian associated with a NetworkX graph G
  Outputs the Hamiltonian in the desired form accepted by QOKit (refer to documentation)
  """
  l = 1
  terms_a = [(-1, tuple([i])) for i in G.nodes()]
  terms_b = [(-int(G.number_of_nodes()), tuple())]
  terms_c = [(l, (int(e[0]), int(e[1]))) for e in G.edges()]
  terms_d = [(l, tuple([int(e[0])])) for e in G.edges()]
  terms_e = [(l, tuple([int(e[1])])) for e in G.edges()]
  terms_f = [(l * int(G.number_of_edges()), tuple())]
  return terms_a + terms_b + terms_c + terms_d + terms_e + terms_f

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
  sim = simclass(N, terms=terms)

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

def train_QAOA(f, N, p):
  """
  Trains the QAOA algorithm
  Input: terms (Hamiltonian)
  Output: res (beta, gamma, final expectation values, and
  information about the classical optimization)
  """

  initial_gamma = -1*np.linspace(0, 1, p)
  initial_beta = np.linspace(1, 0, p)
  input = np.hstack([initial_gamma, initial_beta])

  stored_parameters = []

  def jac(x):
    return Jacobian(f)(x).ravel()

  def callback_function(curr_param):
    stored_parameters.append(curr_param)

  start_time = time.time()
  res = scipy.optimize.minimize(f, input, method='BFGS', jac = jac, callback=callback_function, options={'maxiter': 1000})
  total_time = time.time() - start_time

  return res.nit, res.success, res.fun, stored_parameters, total_time

# TODO: Make sure to change how you're importing graphs
graph_dicts = load_graphs('test (2).pkl')
graph_dicts = graph_dicts[0:10]

# TODO: Make sure to vary values of p
p_list = [3]

df = pd.DataFrame()

graph_loop = tqdm(graph_dicts, desc="Starting...", position=0)

for graph_dict in graph_loop:

  G = nx.Graph(graph_dict)
  terms = get_mis_terms(G)
  adj = nx.to_numpy_array(G)

  N = G.number_of_nodes()

  for p in p_list:

    f = get_qaoa_objective(N, terms=terms, parameterization='theta') # objective function

    graph_loop.set_description(f"Training QAOA (p={p})")
    nit, success, exp, descent_parameters, runtime = train_QAOA(f, N, p) # train QAOA

    graph_loop.set_description(f"Computing metrics (p={p})")

    simclass = qokit.fur.choose_simulator(name='auto') # set up QAOA simulator
    sim = simclass(N, terms=terms)

    diags = sim.get_cost_diagonal() # get Hamiltonian
    ground_truth = np.min(diags) # ground truth from Hamiltonian
    approx = exp / ground_truth # get approximation score

    bin_sols = get_sols(diags, N) # get solutions
    dec_sols = [int(sol, 2) for sol in bin_sols]

    prob_sums = [] # compute sigma_p
    for x in descent_parameters:
      _result = sim.simulate_qaoa(x[:p], x[p:])
      probs = sim.get_probabilities(_result)
      prob_sum = 0
      for i in dec_sols:
        prob_sum += probs[i]
      prob_sums.append(prob_sum)
    prob_sums = np.array(prob_sums)

    if success:
      success = 1
    else:
      success = 0

    descent_parameters = np.vstack(descent_parameters)

    new_row = pd.DataFrame(
        {
            'N': [N],
            'adj': [adj],
            'terms': [terms],
            'p': [p],
            'mixer': ['x'],
            'hamiltonian': [diags],
            'nit': [nit],
            'runtime': [runtime],
            'success': [success],
            'x_descent': [descent_parameters],
            'sigma_p': [prob_sums],
            'approx': [approx],
            }
        )

    df = pd.concat([df, new_row], ignore_index=True)

df.to_csv('toy_data.csv')
df.to_pickle('toy_data.pkl')
