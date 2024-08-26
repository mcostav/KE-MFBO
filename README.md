# KE-MFBO
Knowledge-Enhanced Multi-Fidelity Bayesian Optimisation

We present a novel Multi-Fidelity Bayesian Optimisation framework which builds up on work from Savage et. al. [1] and applies some modifications.
By observing the behaviour of their acquisition function in a multi-fidelity BO code, we implement some important changes which show good performance:
1. Like in Savage et.al. [1], the final iteration is reserved to be made at the highest fidelity (z = 1), but in this case, the multistart takes the three best points at any fidelity and the best point at the highest fidelity, which in some cases shows to improve the last approximation to the global minima.
2. It is observed that some points sampled at a fidelity between 0 and 0.2 are very close to the global optima if evaluated at the highest fidelity. So a random condition is generated so that it is 30% likely that a low-fidelity is selected to be evaluated at the highest fidelity.
3. 
Benchmark tests are carried out on several objective functions of different dimensionalities, taken from Kandasamy et.al [2]:
1. Currin exponential
2. Branin
3. Hartmann 3-D
4. Hartmann 6-D






References:
[1] Tom Savage et al. “Multi-fidelity data-driven design and analysis of reactorand tube simulations”. In: Computers and Chemical Engineering 179 (Nov. 2023). ISSN: 00981354. DOI: 10.1016/j.compchemeng.2023.108410.
[2] Kandasamy expanded paper

