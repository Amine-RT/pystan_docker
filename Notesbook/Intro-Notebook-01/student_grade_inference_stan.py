#! /usr/bin/env python
## student_grade_inference_stan.py

import stan
import numpy as np
import matplotlib.pyplot as plt

# Simulate data
from student_grade_simulation import sample_student_grades
n, p = 30, 5
X, mu, sigma = sample_student_grades(n, p)
sm_data = {'n':n, 'p':p, 'tau':0.5, 'a':1, 'b':0.5, 'X':X}

# Initialise stan object
with open('student_grade_model.stan','r',newline='') as f:
    sm = stan.build(f.read(),sm_data,random_seed=1)

# Select the number of MCMC chains and iterations, then sample
chains, samples, burn = 4, 10000, 1000
fit=sm.sample(num_chains=chains, num_samples=samples, num_warmup=burn, save_warmup=False)

def plot_samples(fit,par,name,true_val=None):
    fig,axs=plt.subplots(2,2,figsize=(10,4),constrained_layout=True)
    fig.canvas.manager.set_window_title('Posterior for '+par)
    for i,j in [(i,j) for i in range(2) for j in range(2)]:
        axs[i,j].autoscale(enable=True, axis='x', tight=True)
    axs[0,0].set_title('Trace plot of log posterior density')
    axs[0,1].set_title('Trace plot of posterior samples of '+name)
    axs[1,0].set_title('Convergence of chain averages for '+name)
    axs[1,1].set_title('Approximate posterior density of '+name)
    par_mx=fit[par].reshape(samples,chains)
    lp_mx=fit['lp__'].reshape(samples,chains)
    for i in range(chains):
        x=i*samples+np.arange(samples)
        axs[0,0].plot(x,lp_mx[:,i])
        axs[0,1].plot(x,par_mx[:,i])
        axs[1,0].plot(x,np.cumsum(par_mx[:,i])/range(1,samples+1))
    axs[1,1].hist(par_mx.flatten(),200, density=True);
    if true_val is not None:
        axs[1,1].axvline(true_val, color='c', lw=2, linestyle='--')
    plt.show()

plot_samples(fit,'mu',r'$\mu$',true_val=mu)
