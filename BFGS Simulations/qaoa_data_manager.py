import numpy as np
import scipy
import networkx as nx
import pickle
import pandas as pd
from tqdm import tqdm
import os
import sklearn
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

import qaoa_opt as qopt
import qaoa_train as qtrain
import qaoa_graph_features as gfeat

def load_graphs(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_sort_graphs(filenames):
    """
    Loads graph dictionaries from a list of pickle files and sorts them by number of nodes.
    Args:
        filenames (list of str): Paths to pickle files containing graph dicts.
    Returns:
        list of dict: Sorted list of graph dictionaries by number of vertices (ascending).
    """
    all_graphs = []

    for filename in filenames:
        with open(filename, 'rb') as f:
            graph_dicts = pickle.load(f)
            all_graphs.extend(graph_dicts)

    # Sort by number of nodes
    sorted_graphs = sorted(all_graphs, key=lambda gd: nx.Graph(gd).number_of_nodes())
    return sorted_graphs

def filter_untested_graphs(graph_dicts, p_list, optimizer_names, result_files=None):
    """
    Filters graph_dicts to include only those not yet tested for all (p, optimizer) combinations.

    Args:
        graph_dicts (list of dict): Graphs to be filtered.
        p_list (list of int): QAOA depths to check.
        optimizer_names (list of str): Optimizers to check.
        result_files (list of str, optional): Explicit list of QAOA result files (CSV/PKL).
            If None, automatically detects in current directory.

    Returns:
        tuple:
            - list of dict: Graphs needing evaluation for at least one (p, optimizer) combo.
            - list of list of int: Untested p values for each graph (same order as output graphs).
            - list of dict: For each output graph, maps p values to list of missing optimizers.
    """
    if result_files is None:
        result_files = [
            f for f in os.listdir('.') if f.endswith('.pkl') and 'QAOA' in f  # Only checks pkl files for efficiency
        ]

    if not result_files:
        print("No QAOA result files found. Returning all input graphs.")
        return graph_dicts, [{p: optimizer_names for p in p_list} for _ in graph_dicts]

    # Load all result files into a single DataFrame
    dfs = []
    for file in result_files:
        try:
            # df = pd.read_csv(file) if file.endswith('.csv') else pd.read_pickle(file)
            df = pd.read_pickle(file)
            if 'adj' in df.columns and 'p' in df.columns and 'optimizer' in df.columns:
                if isinstance(df['adj'].iloc[0], str):
                    df['adj'] = df['adj'].apply(eval)
                df['adj'] = df['adj'].apply(np.array)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")

    if not dfs:
        print("No valid QAOA result data found. Returning all input graphs.")
        return graph_dicts, [{p: optimizer_names for p in p_list} for _ in graph_dicts]

    full_df = pd.concat(dfs, ignore_index=True)

    # Build tested set
    tested = set()
    for _, row in full_df.iterrows():
        key = (row['adj'].tobytes(), int(row['p']), row['optimizer'])
        tested.add(key)

    # Identify untested (p, optimizer) combos per graph
    untested_graphs = []
    untested_opt_dicts = []

    for gd in graph_dicts:
        G = nx.Graph(gd)
        adj = nx.to_numpy_array(G)
        adj_bytes = adj.tobytes()

        missing_opt_by_p = {}

        for p in p_list:
            missing_opts = [opt for opt in optimizer_names if (adj_bytes, p, opt) not in tested]
            if missing_opts:
                missing_opt_by_p[p] = missing_opts

        if missing_opt_by_p:
            untested_graphs.append(gd)
            untested_opt_dicts.append(missing_opt_by_p)

    return untested_graphs, untested_opt_dicts

def generate_output_filenames(graph_dicts, p_list, optimizer_names, input_filenames, prefix='QAOA'):
    """
    Generate descriptive filenames for CSV and pickle output based on graph orders,
    circuit depths, optimizer names, and graph symmetry tags from input filenames.

    Args:
        graph_dicts (list of dict): Graphs used in the experiment.
        p_list (list of int): List of QAOA depths used.
        optimizer_names (list of str): Names of optimizers used.
        input_filenames (list of str): Original input filenames used to load graphs.
        prefix (str): Base prefix for output (default: 'QAOA').

    Returns:
        dict: Filenames for full data and summary in both CSV and PKL formats.
    """
    # --- Extract graph order range ---
    orders = sorted([nx.Graph(gd).number_of_nodes() for gd in graph_dicts])
    min_order = orders[0]
    max_order = orders[-1]
    order_str = f"{min_order}-{max_order}" if min_order != max_order else f"{min_order}"

    # --- Extract circuit depth range ---
    min_p = min(p_list)
    max_p = max(p_list)
    p_str = f"p{min_p}-{max_p}" if min_p != max_p else f"p{min_p}"

    # --- Extract optimizer tag ---
    opt_str = '_'.join(sorted(optimizer_names))
    opt_str = (
        opt_str.replace('layerwise_', 'lw_')
                .replace('adaptive_exploration_', 'ae_')
                .replace('bayesian_', 'bayes_')
                .replace('bfgs_qaoa_', 'bfgs_')
    )

    # --- Detect graph symmetry tags ---
    symmetry_tags = []
    combined_input = ' '.join(input_filenames).lower()
    if 'vertex_and_edge_transitive' in combined_input:
        symmetry_tags.append("Vertex_And_Edge-Transitive")
    if 'arc-transitive' in combined_input:
        symmetry_tags.append("Arc-Transitive")
    if 'vertex-transitive' in combined_input:
        symmetry_tags.append("Vertex-Transitive")

    # --- Construct filename ---
    tag_section = ''.join([f"_{tag}" for tag in symmetry_tags])
    base_name = f"{prefix}{tag_section}_{opt_str}_Order_{order_str}_{p_str}"

    return {
        'base_name': f"{base_name}",
        'csv': f"{base_name}.csv",
        'pkl': f"{base_name}.pkl",
        'summary_csv': f"{base_name}_summary.csv",
        'summary_pkl': f"{base_name}_summary.pkl",
        'complete_csv': f"{base_name}_complete.csv",
        'complete_pkl':  f"{base_name}_complete.pkl",
        'feature_correlations_loss_variance_csv': f"{base_name}_loss_feature_exp_correlations.csv",
        'feature_correlations_loss_variance_pkl': f"{base_name}_loss_feature_exp_correlations.pkl",
        'feature_correlations_mean_grad_variance_csv': f"{base_name}_grad_feature_exp_correlations.csv",
        'feature_correlations_mean_grad_variance_pkl': f"{base_name}_grad_feature_exp_correlations.pkl",
        'graphs': "_".join(input_filenames) + "_features"
    }
    
def save_results_dataframe(df, filenames, desired_append=None):
    """
    Saves the full QAOA training results dataframe to both CSV and pickle formats.

    Args:
        df (pd.DataFrame): Full training results dataframe.
        desired append: desired used to deviate from automated names from input_filenames
    """
    if desired_append is None:
        df.to_csv(filenames['csv'], index=False)
        df.to_pickle(filenames['pkl'])
    
    else:
        df.to_csv(filenames['base_name'] + desired_append + '.csv', index=False)
        df.to_pickle(filenames['base_name'] + desired_append + '.pkl')

def save_summary_statistics(df, filenames):
    """
    Computes and saves summary statistics of QAOA training results for each p and optimizer,
    including exponential fit via linear regression on log(mean_grad_var) and log(loss_var).
    """
    summary_rows = []

    for opt_name in df['optimizer'].unique():
        df_opt = df[df['optimizer'] == opt_name]

        for p_val in sorted(df_opt['p'].unique()):
            df_p = df_opt[df_opt['p'] == p_val]

            row = {
                'optimizer': opt_name,
                'p': p_val,

                'avg_nit': df_p['nit'].mean(),
                'var_nit': df_p['nit'].var(),
                'min_nit': df_p['nit'].min(),
                'max_nit': df_p['nit'].max(),

                'avg_f_calls': df_p['f_calls'].mean(),
                'var_f_calls': df_p['f_calls'].var(),
                'min_f_calls': df_p['f_calls'].min(),
                'max_f_calls': df_p['f_calls'].max(),

                'avg_approx': df_p['approx'].mean(),
                'var_approx': df_p['approx'].var(),
                'min_approx': df_p['approx'].min(),
                'max_approx': df_p['approx'].max(),

                'avg_percent_error': df_p['percent_error'].mean(),
                'var_percent_error': df_p['percent_error'].var(),
                'min_percent_error': df_p['percent_error'].min(),
                'max_percent_error': df_p['percent_error'].max(),
            }

            sizes = df_p['graph'].apply(lambda gd: nx.Graph(gd).number_of_nodes())

            for regvar in ['mean_grad_variance', 'loss_variance']:
                if regvar in df_p.columns:
                    try:
                        variances = df_p[regvar]
                        valid = (variances > 0) & (~sizes.isna())

                        if valid.sum() >= 2:
                            x = sizes[valid].values
                            y = variances[valid].values
                            log_y = np.log(y)

                            slope, intercept, r, _, _ = linregress(x, log_y)

                            row['avg_' + regvar] = y.mean()
                            row['exp_fit_a' +  '_' + regvar] = np.exp(intercept)
                            row['exp_fit_b' +  '_' + regvar] = slope
                            row['pearson_corr_exp_fit' +  '_' + regvar] = r
                        else:
                            row['avg_' + regvar] = None
                            row['exp_fit_a' + '_' + regvar] = None
                            row['exp_fit_b' + '_' + regvar] = None
                            row['pearson_corr_exp_fit' + '_' + regvar] = None

                    except Exception:
                        row['avg_' + regvar] = None
                        row['exp_fit_a' +  '_' + regvar] = None
                        row['exp_fit_b' +  '_' + regvar] = None
                        row['pearson_corr_exp_fit' +  '_' + regvar] = None
                        
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=['optimizer', 'p']).reset_index(drop=True)
    summary_df['rank_by_f_calls'] = summary_df.groupby('p')['avg_f_calls'].rank(method='min').astype(int)
    summary_df.to_csv(filenames['summary_csv'], index=False)
    summary_df.to_pickle(filenames['summary_pkl'])
    
    
def save_complete_results(qaoa_df, graph_dicts, outnames):
    # Compute and merge graph features
    gfeatures = gfeat.compute_and_save_graph_features(graph_dicts, outnames['graphs'])
    merged = gfeat.merge_on_adjacency(qaoa_df, gfeatures)

    # Save merged dataframe
    merged.to_csv(outnames['complete_csv'], index=False)
    merged.to_pickle(outnames['complete_pkl'])

    # All feature names (excluding metadata columns)
    feature_cols = [col for col in gfeatures.columns if col not in ['adj', 'adj_key', 'Number of vertices', 'N']]
    regvars = ['mean_grad_variance', 'loss_variance']
    results_dict = {var: [] for var in regvars}

    # Estimate total number of iterations for outer tqdm
    total_jobs = sum(
        len(merged[merged["optimizer"] == optimizer]["p"].unique()) 
        for optimizer in df['optimizer'].unique()
    )

    outer_loop = tqdm(total=total_jobs, desc="Analyzing optimizers/p", position=0)

    for opt_name in merged['optimizer'].unique():
        df_opt = merged[merged['optimizer'] == opt_name]

        for p_val in df_opt["p"].unique():
            df_p = df_opt[df_opt["p"] == p_val]

            inner_loop = tqdm(
                total=len(regvars) * len(feature_cols),
                desc=f"{optimizer}, p={p_val}",
                leave=False,
                position=1
            )

            for regvar in regvars:
                for feat in feature_cols:
                    results_df = analyze_feature_regressions(
                        df_p,
                        feat,
                        regvar,
                        min_cluster_size=max(5, len(df_p) // 20),
                        min_samples=max(2, len(df_p) // 40)
                    )

                    if results_df is not None and not results_df.empty:
                        results_df = results_df.copy()
                        results_df['optimizer'] = optimizer
                        results_df['p'] = p_val
                        results_dict[regvar].append(results_df)

                    inner_loop.update(1)

            inner_loop.close()
            outer_loop.update(1)

    outer_loop.close()

    for regvar in regvars:
        if results_dict[regvar]:
            corr_df = pd.concat(results_dict[regvar], ignore_index=True)
            corr_df = corr_df.sort_values(by='pearson_r_log').reset_index(drop=True)

            # Save correlation analysis
            corr_df.to_csv(outnames['feature_correlations_' + regvar + '_csv'], index=False)
            corr_df.to_pickle(outnames['feature_correlations_' + regvar + '_pkl'])
        else:
            print("No correlation results were produced for " + regvar + ".")
    
def analyze_feature_regressions(
    df,
    feature,
    regvar,
    min_cluster_size=8,
    min_samples=None,
    min_points=4
):
    """
    Perform linear regression of log(mean_grad_variance) vs N
    for each value of a discrete feature or each cluster in a continuous feature,
    segregated by optimizer and QAOA depth p. Saves the adjacency matrices used.
    """
    results = []

    is_integer = pd.api.types.is_integer_dtype(df[feature]) or all(
        df[feature].dropna() == df[feature].dropna().astype(int)
    )

    for optimizer in tqdm(df['optimizer'].unique(), desc=f"Optimizers ({feature})"):
        df_opt = df[df['optimizer'] == optimizer]
        
        for p_val in sorted(df_opt['p'].unique()):
            df_p = df_opt[df_opt['p'] == p_val]

            if is_integer:
                grouped = df_p.groupby(feature)
                for feat_val, group in grouped:
                    if len(group) < min_points or any(group[regvar] <= 0):
                        continue
                    log_y = np.log(group[regvar].values)
                    slope, intercept, r, _, _ = linregress(group['N'].values, log_y)

                    results.append({
                        "feature": feature,
                        "feature_value": feat_val,
                        "feature_type": "discrete",
                        "cluster_label": f"val_{feat_val}",
                        "optimizer": optimizer,
                        "p": p_val,
                        "pearson_r_log": r,
                        "exp_a": np.exp(intercept),
                        "exp_b": slope,
                        "min_val": feat_val,
                        "max_val": feat_val,
                        "mean_val": feat_val,
                        "n_points": len(group),
                        "adjs": group["adj"].tolist()
                    })

            else:
                if len(df_p) < min_cluster_size:
                    continue

                X_scaled = StandardScaler().fit_transform(df_p[[feature]])
                hdb = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples or min_cluster_size // 2)
                labels = hdb.fit_predict(X_scaled)

                df_p = df_p.copy()
                df_p['cluster'] = labels

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue  # skip noise

                    group = df_p[df_p['cluster'] == cluster_id]
                    if len(group) < min_points or any(group[regvar] <= 0):
                        continue

                    log_y = np.log(group[regvar].values)
                    slope, intercept, r, _, _ = linregress(group['N'].values, log_y)

                    results.append({
                        "feature": feature,
                        "feature_value": None,
                        "feature_type": "continuous",
                        "cluster_label": f"cluster_{cluster_id}",
                        "optimizer": optimizer,
                        "p": p_val,
                        "pearson_r_log": r,
                        "exp_a": np.exp(intercept),
                        "exp_b": slope,
                        "min_val": group[feature].min(),
                        "max_val": group[feature].max(),
                        "mean_val": group[feature].mean(),
                        "n_points": len(group),
                        "adjs": group["adj"].tolist()
                    })

    return pd.DataFrame.from_records(results)
