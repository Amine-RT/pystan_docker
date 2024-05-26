// student_grade_model.stan

data {
    int<lower=0> n; // number of students
    int<lower=0> p; // number of tests
    array[n, p] int<lower=0, upper=100> X; // student test grades
    real<lower=0> tau;
    real<lower=0> a;
    real<lower=0> b;
}
parameters {
    array[n] real z;
    real mu;
    real<lower=0> sigma_sq;
}
transformed parameters {
    array[n] real<lower=0, upper=1> theta;
    real sigma;
    theta = inv_logit(z);
    sigma = sqrt(sigma_sq);
}
model {
    sigma_sq ~ inv_gamma(a,b);
    mu ~ normal(0, sigma * tau);
    z ~ normal(mu, sigma);
    for (i in 1:n)
      X[i] ~ binomial(100,theta[i]);
}
