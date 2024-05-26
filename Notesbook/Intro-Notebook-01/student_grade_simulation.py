#! /usr/bin/env python
## student_grade_simulation.py

import numpy as np
gen = np.random.default_rng(seed=0)

def sample_student_grades(n=30,p=5): #n students, p tests
    mu,sigma=sample_student_grade_parameters()
    z=gen.normal(mu,sigma,size=n)
    X=[gen.binomial(100,1/(1.0+np.exp(-z_i)),size=p) for z_i in z]
    return(X,mu,sigma)

def sample_student_grade_parameters(a=1,b=.5,m=0,tau=.5):
    sigma=1.0/np.sqrt(gen.gamma(a,1.0/b))
    mu=gen.normal(m,tau*sigma)
    return(mu,sigma)

#print(np.array(sample_student_grades()))
