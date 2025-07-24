def jac(f, x, cur_calls=0, track_calls=True, mthd='central'):
    if track_calls:
            cur_calls += (1 + int(mthd=='central')) * len(x) # Note: complex difference unsupported
            return nd.Gradient(f)(x), cur_calls

    else:
        return nd.Gradient(f, method=mthd)(x)

with open("toy_data.pkl", 'rb') as f:
  df = pickle.load(f)

n_samples = 1000

loss_var = []
grad_var = []

for i in range(len(df)):

  N = int(df["N"][i])
  terms = df["terms"][i]
  p = int(df["p"][i])
  f = get_qaoa_objective(N, terms=terms, parameterization='theta')

  # initialize jac / hes list / array
  loss_evaluations = []
  grad_evaluations = []
  
  for j in range(n_samples):
    param = 2 * np.pi * np.random.rand(2 * p)
    
    # evalute jac / hes and append
    loss_evaluations.append(f(param))
    grad_evaluations.append(jac(f, param, track_calls=False))

  loss_var.append(np.var(loss_evaluations,ddof=1))
  grad_var.append(np.mean(np.var(grad_evaluations,ddof=1,axis=0)))

df['loss_var'] = loss_var
df['grad_var'] = grad_var

df.to_csv('test.csv')
df.to_pickle('test.pkl')

