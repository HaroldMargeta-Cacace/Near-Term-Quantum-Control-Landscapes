import numpy as np
import scipy
import networkx as nx
import pickle
import pandas as pd
from tqdm import tqdm

# Helper functions

feature_names = {
0:'Number of vertices',
1:'Number of edges',
2:'Edge density',
3:'Mean degree',
4:'Standard deviation of degrees',
5:'Skewness of degrees',
6:'Minimum degree',
7:'Maximum degree',
8:'Diameter',
9:'Radius',
10: 'Vertex connectivity',
11: 'Edge connectivity',
12: 'Global clustering coefficient',
13: 'Mean local clustering coefficient',
14: 'Standard deviation of local clustering coefficients',
15: 'Skewness of local clustering coefficients',
16: 'Minimum local clustering coefficient',
17: 'Maximum local clustering coefficient',
18: 'Treewidth',
19: 'Average path length',
20: 'Circuit rank',
21: 'Girth',
22: 'Mean betweenness centrality',
23: 'Standard deviation of betweenness centralities',
24: 'Skewness of betweenness centralities',
25: 'Minimum betweenness centrality',
26: 'Maximum betweenness centrality',
27: 'Algebraic connectivity',
28: 'Von Neumann entropy',
29: 'Adjacency spectrum mean',
30: 'Adjacency spectrum standard deviation',
31: 'Adjacency spectrum skewness',
32: 'Adjacency spectrum min',
33: 'Adjacency spectrum max',
34: 'Laplacian spectrum mean',
35: 'Laplacian spectrum standard deviation',
36: 'Laplacian spectrum skewness',
37: 'Laplacian spectrum min',
38: 'Laplacian spectrum max',
39: 'Planarity',
40: 'Mean harmonic centrality',
41: 'Standard deviation of harmonic centralities',
42: 'Skewness of harmonic centralities',
43: 'Minimum harmonic centrality',
44: 'Maximum harmonic centrality',
45: 'Harmonic diameter',
46: 'Mean core number',
47: 'Standard deviation of core numbers',
48: 'Skewness of core numbers',
49: 'Minimum core number',
50: 'Maximum core number',
51: 'Chordality',
52: 'Haemers bound',
53: 'Claw-free'
}

def flatten(x):
    if not isinstance(x, list):
        return [x]
    else:
        return [z for y in x for z in flatten(y)]

def skew1(input_list):
    if abs(max(input_list) - min(input_list)) < 0.000000001:
        return 0
    return sp.stats.skew(input_list)

def statistics(input_list, **kwargs):
    output = []
    if kwargs.get('include_mean', True):
        output.append(np.mean(input_list))
    if kwargs.get('include_std', True):
        output.append(np.std(input_list))
    if kwargs.get('include_skew', True):
        output.append(skew1(input_list))
    if kwargs.get('include_min', True):
        output.append(min(input_list))
    if kwargs.get('include_max', True):
        output.append(max(input_list))
    return output

# Note transitivity is also known as the global clustering coefficient

def num_vertices(G):
    return G.number_of_nodes()

def num_edges(G):
    return G.number_of_edges()

def degree_statistics(G):
    degrees = [deg for _, deg in G.degree()]
    return statistics(degrees)

def diameter(G):
    diameter_sum = 0
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        diameter_sum += nx.diameter(C)
    return diameter_sum

def radius(G):
    radius_sum = 0
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        radius_sum += nx.radius(C)
    return radius_sum

def treewidth_approx(G):
    return approximation.treewidth_min_fill_in(G)[0]

def clustering_statistics(G):
    clustering_coeffs = list(nx.clustering(G).values())
    return statistics(clustering_coeffs)

def average_shortest_path_length(G):
    if G.number_of_nodes() < 2:
        return 0
    avg = 0
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        n = C.number_of_nodes()
        avg += nx.average_shortest_path_length(C) * n * (n - 1)
    avg /= (G.number_of_nodes() * (G.number_of_nodes() - 1))
    return avg

def circuit_rank(G):
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

def girth(G):
    g = nx.girth(G)
    return 0 if g == math.inf else g

def betweenness_centrality_statistics(G):
    bc = list(nx.betweenness_centrality(G).values())
    return statistics(bc)

def algebraic_connectivity(G):
    if G.number_of_nodes() < 2:
        return 0
    return nx.algebraic_connectivity(G, method='tracemin_lu')

def von_neumann_entropy(G):
    degree_sum = sum(deg for _, deg in G.degree())
    if degree_sum == 0:
        return 0
    rho = (1 / degree_sum) * nx.laplacian_matrix(G)
    rho = rho.todense()
    evals = np.linalg.eigvals(rho)
    entropy = -sum(
        np.real(e) * math.log2(np.real(e))
        for e in evals if np.real(e) > 0
    )
    return entropy

def adjacency_spectrum_statistics(G):
    evals = [np.real(ev) for ev in nx.adjacency_spectrum(G)]
    return statistics(evals)

def laplacian_spectrum_statistics(G):
    evals = nx.laplacian_spectrum(G)
    return statistics(evals)

def planarity(G):
    return int(nx.is_planar(G))

def harmonic_centrality_statistics(G):
    return statistics(list(nx.harmonic_centrality(G).values()))

def harmonic_diameter(G):
    result = nx.harmonic_diameter(G)
    return 0 if result == math.inf or G.number_of_nodes() < 2 else result

def core_number_statistics(G):
    return statistics(list(nx.core_number(G).values()))

def chordality(G):
    return int(nx.is_chordal(G))

def haemers_bound(G):
    n = G.number_of_nodes()
    degrees = [deg for _, deg in G.degree()]
    min_degree = min(degrees)
    evals = [np.real(ev) for ev in nx.adjacency_spectrum(G)]
    min_eval = min(evals)
    max_eval = max(evals)
    denom = min_degree ** 2 - max_eval * min_eval
    if abs(denom) < 1e-12:
        return n
    return (-n * max_eval * min_eval) / denom

def claw_free(G):
    H = nx.complete_bipartite_graph(1, 3)
    isomatcher = nx.isomorphism.GraphMatcher(G, H)
    return int(not isomatcher.subgraph_is_isomorphic())

def graph_features(G):
    graph_features_unflattened = [
        num_vertices, num_edges, nx.density, degree_statistics, diameter,
        radius, nx.node_connectivity, nx.edge_connectivity, nx.transitivity,
        clustering_statistics, treewidth_approx, average_shortest_path_length,
        circuit_rank, girth, betweenness_centrality_statistics, algebraic_connectivity,
        von_neumann_entropy, adjacency_spectrum_statistics, laplacian_spectrum_statistics,
        planarity, harmonic_centrality_statistics, harmonic_diameter,
        core_number_statistics, chordality, haemers_bound, claw_free
    ]
    
    return flatten([f(G) for f in graph_features_unflattened])

def compute_and_save_graph_features(graph_dict, output_file="graph_features"):
    """
    Efficiently computes and saves graph features using column-wise construction.
    """
    adjs = []
    adj_keys = []
    feature_data = {}

    first_pass = True
    for _, G in tqdm(graph_dict.items(), desc="Computing graph features"):
        A = nx.to_numpy_array(G, dtype=int)
        adjs.append(A)
        adj_keys.append(tuple(map(tuple, A)))

        features = graph_features(G)

        # Determine how to unpack and initialize feature columns
        if isinstance(features, dict):
            if first_pass:
                for key in features.keys():
                    feature_data[key] = []
                first_pass = False

            for key, val in features.items():
                feature_data[key].append(val)

        elif isinstance(features, tuple) and len(features) == 2:
            values, names = features
            if first_pass:
                for name in names:
                    feature_data[name] = []
                first_pass = False

            for name, val in zip(names, values):
                feature_data[name].append(val)

        else:
            raise ValueError("graph_features must return a dict or (values, names) tuple.")

    # Assemble final DataFrame
    df_dict = {
        "adj": adjs,
        "adj_key": adj_keys,
        **feature_data
    }
    df = pd.DataFrame(df_dict)

    df.to_csv(output_file + ".csv", index=False)
    df.to_pickle(output_file + ".pkl", index=False)

    return df

def add_adj_merge_key(df, adj_col='adj', key_col='adj_key'):
    df = df.copy()
    df[key_col] = df[adj_col].apply(lambda a: tuple(map(tuple, a)))
    return df

def merge_on_adjacency(qaoa_df, graph_features_df, adj_col='adj'):
    # Add merge keys to qaoa dataframe
    qaoa_df1 = add_adj_merge_key(qaoa_df, adj_col=adj_col)

    # Find common columns except the merge key
    common_cols = set(qaoa_df1.columns).intersection(graph_features_df.columns)
    common_cols.discard('adj_key')  # keep the key column for merging

    # Drop overlapping columns from the right dataframe
    graph_features_df_reduced = graph_features_df.drop(columns=common_cols)

    # Merge on the hashable adjacency key
    merged_df = pd.merge(
        qaoa_df1,
        graph_features_df_reduced,
        on='adj_key',
        how='left'
    )

    # Remove the helper key column after merge
    merged_df = merged_df.drop(columns=['adj_key', 'Number of vertices'])
    return merged_df