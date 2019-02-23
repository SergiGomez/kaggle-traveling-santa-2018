# Traveling Santa 2018 - Prime Paths

This is a version of the "Traveling Salesman Problem"

Link to the Kaggle challenge www.kaggle.com/c/quora-insincere-questions-classification

## Overview
We are provided a list of cities and their coordinates in cities.csv. We then need to create the shortest possible path that visits all those cities. Paths have the following constraints:
- Paths must start and end at the North Pole (CityId = 0)
- We must visit every city exactly once
- The distance between two paths is the 2D Euclidean distance. However, every 10th step (stepNumber % 10 == 0) there is a penalty of the form of a 10% more lengthy distance, unless coming from a prime CityId.

## Solution
The approach proposed executes the LKH algorithm (http://akira.ruc.dk/~keld/research/LKH/) using the excellent solver made by the same creators of the algorithm. On top of the path calculated by the solver, that is, an initial solution, we then apply some other algorithms that try to reduce the distance proposed by LKH.

This approach made us be ranked us top 5% (we were awarded silver metal) and it was an excellent opportunity to solve a NP-complete problem.

  
