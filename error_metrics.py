import qokit

import networkx as nx
import pickle
import pandas as pd
import numpy as np
import qaoa_data_manager as qmngr

filename = "/content/QAOA_bfgs_qaoa_Order_8-20_p1-10.pkl"
with open(filename, 'rb') as f:
  df = pickle.load(f)

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

simclass = qokit.fur.choose_simulator(name='auto') # set up QAOA simulator

approx_list = []
graph_dicts = []
p_list = []
optimizer_list = []


old_adj = np.empty(1)

for i in range(len(df)):
  adj = df["adj"][i]
  exp = df["f_vals"][i][-1]
  p = df["p"][i]
  optimizer = df["optimizer"][i]

  if p not in p_list:
    p_list.append(p)

  if optimizer not in optimizer_list:
    optimizer_list.append(optimizer)

  if not np.array_equal(adj, old_adj):
    G = nx.from_numpy_array(adj)
    graph_dict = nx.to_dict_of_lists(G)
    N = G.number_of_nodes()
    terms = get_mis_terms(G)

    sim = simclass(N, terms=terms)

    diags = sim.get_cost_diagonal() # get Hamiltonian
    ground_truth = np.min(diags) # ground truth from Hamiltonian
    approx = exp / ground_truth

    graph_dicts.append(graph_dict)

  else:
    approx = exp / ground_truth

  old_adj = adj

  approx_list.append(approx)

df['approx'] = approx_list

filenames = qmngr.generate_output_filenames(graph_dicts, p_list, optimizer_list, ["a"])

qmngr.save_results_dataframe(df, filenames, "error_metrics")
