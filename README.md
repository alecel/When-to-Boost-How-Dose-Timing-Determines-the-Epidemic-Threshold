# When to Boost: How Dose Timing Determines the Epidemic Threshold

The repository contains code and data used in the paper [When to Boost: How Dose Timing Determines the Epidemic Threshold](https://arxiv.org/abs/2502.16715). 

The pickle files contain the network analyzed in the paper, they are stored as [NetworkX](https://networkx.org) graphs. We analyzed three different networks: i) an Erdős-Rényi-Gilbert(ER) graph; ii) the [Enron](https://snap.stanford.edu/data/email-Enron.html) email network, and iii) a geometric random graph generated using the $\mathbb{S}^1$ [model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.078701).

The file P_qs.py is used to run the simulations, we used [EoN](https://epidemicsonnetworks.readthedocs.io/en/latest/index.html) (Epidemics on Networks) a Python module designed to study infectious processes spreading in networks. Specifically, we used the Gillespie implementation provided by EoN and the NetworkX package to implement the $S^3I^2$ model. The model is implemented as a directed NetworkX graph, and is passed as argument to the Gillespie algorithm.  



