## Statistics Explorations

This repository provides a collection of Python programs that can be used
to explore concepts in Statistics and Probability Theory. 

### Requirements
- Python 3.7
- Plotly 4.3.0
- Pandas
- Scipy

### Installation
- Install the requirements preferably in a virtual (or conda) environment
- Download the repository files and run the programs

### Programs

#### binomial.py
Plots the probability mass function (PMF) of a Binomial distribution 
using two methods: (1) using the 
explicit PMF formula, and (2) simulating 10,000 experiments and using the 
results to plot an estimated PMF. 

Generates charts displayed in a browser using the modern plotly library

Usage:

`python binomial.py -n <value of n> -p <value of p>`

where `p` and `n` correspond to the parameters of the Binomial 
distribution. This distribution defines the probability 
of having `k` successes in `n` trials. The outcome of each trial is either
a success with probability `p` or a failure with probability `1-p`. 

