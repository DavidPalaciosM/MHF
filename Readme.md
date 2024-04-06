# Metaheuristic Firebreak Generation
## David Palacios
Main program 

# Requirements
- Cellfire Simulator (with compilation already done: instructions on blabla.com)
- Python 3.6
- Libraries from requirements.txt

# Usage
For the execution of metaheuristics, execute main_heuristics.py. By default, both GRASP and GA will be executed. If only one is desired, modify said file by not calling the respective module.
Some arguments that can be changed are:

- Maximum execution time: by default set to 7200 seconds (2 hours), can be changed on the variable maxTime in main_heuristics.py

- Number of tests: by default set to 5, can be changed of the length of seeds variable in main_heuristics.py

- Landscape: By default using Exp100x100_hetero_C1, more can be found on Cell2Fire Github.

- Percentage of treatment: by default using different percentages from 1% until 15%. If only some of them of different are required, modify list of percentages in main_heuristics.py

-For other details, contact david.palacios@ug.uchile.cl

# Results 
Results will be generated in the results folder, by the name of grasp_stats_XX (or ga_stats_XX), where XX represents the number of the simulation set in the number of tests
