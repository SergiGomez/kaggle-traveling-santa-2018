#de momento no entra en uso ya que es una librería que da problemas, y que devuelve resultados erroneos

# libraries

from concorde.tsp import TSPSolver
import numpy as np
import pandas as pd
import time


# read files

cities = pd.read_csv('./input/cities.csv')
cities.X = cities.X * 1000
cities.Y = cities.Y * 1000

# concorde solver

solver = TSPSolver.from_data(
    cities.X,
    cities.Y,
    norm="EUC_2D"
)

t = time.time()
tour_data = solver.solve(verbose = True, random_seed = 92) # solve() doesn't seem to respect time_bound for certain values?
print(time.time() - t)
print(tour_data.found_tour)

# submission

pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('submission2.csv', index=False)
