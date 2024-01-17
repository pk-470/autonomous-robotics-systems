from functions import *
from ga_min import *


CONFIG = {
    # Initialization
    "function": rastrigin,
    "organisms_num": 50,
    # Genetic algorithm (options: 1, 2)
    "mode": 1,
}


ga = GA_Min(CONFIG)
ga.genetic_algorithm(generations=400, animate=True)
