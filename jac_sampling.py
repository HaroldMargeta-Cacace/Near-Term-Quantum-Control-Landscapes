with open("toy_data.pkl", 'rb') as f:
  df = pickle.load(f)

n_samples = 1000

for i in range(len(df)):

  N = int(df["N"][i])
  terms = df["terms"][i]
  p = int(df["p"][i])
  f = get_qaoa_objective(N, terms=terms, parameterization='theta')

  # initialize jac / hes list / array

  for j in range(n_samples):
    param = 2 * np.pi * np.random.rand(2 * p)
    
    # evalute jac / hes and append
