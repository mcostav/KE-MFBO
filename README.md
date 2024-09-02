# Trajectory Informed Multi-fidelity Bayesian Optimisation with Increased High-Fidelity Sampling (TIMBO-IHFS)

This repository contains the implementation of a novel multi-fidelity Bayesian optimization strategy, derived from the acquisition function and algorithm developed by Savage et. al. [1].

## Overview

We present a novel Multi-Fidelity Bayesian Optimisation framework which effectively utilises the current dataset information and leverages cheap approximations to improve optimisation efficiency.
By observing the behaviour of the cost-adjusted UCB acquisition function developed by Savage et.al. [1], we implement some important changes which show good performance:

  1. Like in Savage et.al. [1], the final iteration is reserved to be made at the highest fidelity (z = 1), but in this case, the multistart takes the three best points at any fidelity and the best point at the highest fidelity, which in some cases shows to improve the last approximation to the global minima.
  2. It is observed that some points sampled at a fidelity between (0 and 0.2) and others between (0.8 and 1) are very close to the global optima if evaluated at the highest fidelity. So a random condition is generated so that it is 20% likely that a low-fidelity is selected to be evaluated at the highest fidelity and 80% likely that a high fidelity is evaluated at the highest fidelity.

## Cost-adjusted acquisition function and greedy acquisition function
![image](https://github.com/user-attachments/assets/9982fde7-953b-4310-80d9-e7d81acc0ee9)
![image](https://github.com/user-attachments/assets/1826992d-11e5-4555-8687-a1a288e3c86a)


Benchmark tests are carried out on several objective functions of different dimensionalities, taken from Kandasamy et.al [2]:
1. Currin exponential
2. Branin
3. Hartmann 3-D
4. Hartmann 6-D

Three cost functions:
1. (1 + z)
2. (1 + 5z)
3. (1 + 5z)**2

## Results (Trajectory plots)
The order of the images being:
1. Currin exponential for cost_function 1 and 2
2. Branin for cost function 1 and 2
![objective_function_1_cost_function_1_combined_plot](https://github.com/user-attachments/assets/47599cc5-c3e2-452e-a136-42d2b32c14b9)![objective_function_1_cost_function_2_combined_plot](https://github.com/user-attachments/assets/0e26984f-1be9-4c6c-a31b-2539535efcec)!![objective_function_2_cost_function_2_combined_plot](https://github.com/user-attachments/assets/ed61d2b1-5e64-4118-8400-cd8d3f08eb11)
[objective_function_2_cost_function_1_combined_plot](https://github.com/user-attachments/assets/9960480e-d51b-4f69-ab13-c65542eeff97)

## Results (Convergence plots)
The order of the images being:
1. Currin exponential for cost_function 1 and 2
2. Branin for cost function 1 and 2


![objective_function_1_cost_function_1_log_regret_all_fidelities](https://github.com/user-attachments/assets/c7ba1e77-2575-46dd-b8b9-43568cf75972)!![objective_function_2_cost_function_2_log_regret_all_fidelities](https://github.com/user-attachments/assets/213af2f6-f469-4920-b783-6a2e3faf8d4a)
[objective_function_1_cost_function_2_log_regret_all_fidelities](https://github.com/user-attachments/assets/09f513db-7f00-4812-839f-a1758eea6993)![objective_function_2_cost_function_1_log_regret_all_fidelities](https://github.com/user-attachments/assets/27146092-88d0-4d6a-80b0-6b77ca4b9e28)






# References:
[1] Tom Savage et al. “Multi-fidelity data-driven design and analysis of reactorand tube simulations”. In: Computers and Chemical Engineering 179 (Nov. 2023). ISSN: 00981354. DOI: 10.1016/j.compchemeng.2023.108410.

[2] Kirthevasan Kandasamy et al. “Multi-fidelity Bayesian Optimisation with Con-tinuous Approximations”. In: (Mar. 2017).

