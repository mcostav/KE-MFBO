# Knowledge-Enhanced Multi-Fidelity Bayesian Optimisation (KE-MFBO)

This repository contains the implementation of a novel multi-fidelity Bayesian optimization strategy, derived from the acquisition function and algorithm developed by Savage et. al. [1].

## Overview

We present a novel Multi-Fidelity Bayesian Optimisation framework which effectively utilises the current dataset information and leverages cheap approximations to improve optimisation efficiency.
By observing the behaviour of the cost-adjusted UCB acquisition function developed by Savage et.al. [1], we implement some important changes which show good performance:

  1. Like in Savage et.al. [1], the final iteration is reserved to be made at the highest fidelity (z = 1), but in this case, the multistart takes the three best points at any fidelity and the best point at the highest fidelity, which in some cases shows to improve the last approximation to the global minima.
  2. It is observed that some points sampled at a fidelity between 0 and 0.2 are very close to the global optima if evaluated at the highest fidelity. So a random condition is generated so that it is 30% likely that a low-fidelity is selected to be evaluated at the highest fidelity.
  3. BLABLABLA

# Cost-adjusted acquisition function and greedy acquisition function
![image](https://github.com/user-attachments/assets/9982fde7-953b-4310-80d9-e7d81acc0ee9)
![image](https://github.com/user-attachments/assets/1826992d-11e5-4555-8687-a1a288e3c86a)


Benchmark tests are carried out on several objective functions of different dimensionalities, taken from Kandasamy et.al [2]:
1. Currin exponential
2. Branin
3. Hartmann 3-D
4. Hartmann 6-D

## Key Equations

The multi-fidelity Bayesian optimization can be formulated as:

$$
\text{minimize} \; f(x) = \mathbb{E}[f(x, s)] \quad \text{subject to} \; x \in \mathcal{X}, s \in \mathcal{S}
$$

where $x$ represents the input parameters and $s$ denotes different fidelity levels.





References:
[1] Tom Savage et al. “Multi-fidelity data-driven design and analysis of reactorand tube simulations”. In: Computers and Chemical Engineering 179 (Nov. 2023). ISSN: 00981354. DOI: 10.1016/j.compchemeng.2023.108410.
[2] Kirthevasan Kandasamy et al. “Multi-fidelity Bayesian Optimisation with Con-tinuous Approximations”. In: (Mar. 2017).

