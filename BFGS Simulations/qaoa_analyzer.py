import numpy as np
import scipy
import networkx as nx
import pickle
import pandas as pd
from tqdm import tqdm
import time

import numdifftools as nd
import qokit
from qokit import get_qaoa_objective
import qaoa_opt as qopt
import qaoa_train as qtrain
import qaoa_data_manager as qmngr
import qaoa_graph_features as gfeat

graph_dicts = qmngr.load_and_sort_graphs(['Dataset_Up_To_Order_7_a','Dataset_Up_To_Order_7_b','Dataset_Up_To_Order_7_c'])
p_list = [1, 2, 3, 4, 5, 6, 7]
optimizer_names = ['bfgs_qaoa']
df = pd.read_pickle('QAOA_bfgs_Order_1-7_p1-7.pkl')

# SAVE
filelist = ['Dataset_Up_To_Order_7_a','Dataset_Up_To_Order_7_b','Dataset_Up_To_Order_7_c']
qmngr.save_summary_statistics(df, graph_dicts, optimizer_names, p_list, filelist)
qmngr.save_complete_results(df, graph_dicts, p_list, optimizer_names, filelist)
