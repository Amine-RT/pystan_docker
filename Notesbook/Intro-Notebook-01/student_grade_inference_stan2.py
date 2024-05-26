#! /usr/bin/env python
## student_grade_inference_stan.py

import pystan
import numpy as np
import matplotlib.pyplot as plt
import student_grade_simulation_old

# Initialise stan object
sgm = pystan.StanModel(file='student_grade_model_old.stan')

# Simulate data
n, p = 30, 5
X, mu, sigma = student_grade_simulation_old.sample_student_grades(n, p)
sgm_data = {'n':n, 'p':p, 'm':0, 'tau':0.5, 'a':1, 'b':0.5, 'X':X}

# Select the number of MCMC chains and iterations, then sample
chains, samples, burn = 4, 10000, 1000
fit=sgm.sampling(data=sgm_data,chains=chains,iter=samples+burn,warmup=burn)

# Plot sampling output for one of the parameters
def plot_samples(fit,par,name,true_val=None):
    f=fit.extract(pars=(par,'lp__'),permuted=False)
    samples,chains=f[par].shape
    fig,axs=plt.subplots(2,2,figsize=(10,4),constrained_layout=True)
    fig.canvas.manager.set_window_title('Posterior for '+par)
    for i,j in [(i,j) for i in range(2) for j in range(2)]:
        axs[i,j].autoscale(enable=True, axis='x', tight=True)
    axs[0,0].set_title('Trace plot of log posterior density')
    axs[0,1].set_title('Trace plot of posterior samples of '+name)
    axs[1,0].set_title('Convergence of chain averages for '+name)
    axs[1,1].set_title('Approximate posterior density of '+name)
    for i in range(chains):
        x=i*samples+np.arange(samples)
        axs[0,0].plot(x,f['lp__'][:,i])
        axs[0,1].plot(x,f[par][:,i])
        axs[1,0].plot(x,np.cumsum(f[par][:,i])/range(1,samples+1))
    axs[1,1].hist(fit[par],200, density=True);
    if true_val is not None:
        axs[1,1].axvline(true_val, color='c', lw=2, linestyle='--')
    plt.show()

plot_samples(fit,'mu',r'$\mu$',true_val=mu)
