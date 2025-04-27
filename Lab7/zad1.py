import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import math
import numpy as np  
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
x_max = [1,1,1,1,1,1]
x_min = [0,0,0,0,0,0]
my_bounds = (x_min, x_max)

def endurance(params):
    return math.exp(-2*(params[1]-math.sin(params[0]))**2)+math.sin(params[2]*params[3])+math.cos(params[4]*params[5])


def f(params):
    
    n_particles = params.shape[0]
    j = [endurance(params[i]) for i in range(n_particles)]
    
    
    return -1 * np.array(j)


optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

cost, pos = optimizer.optimize(f, iters=1000)


print(f"Best cost: {-cost}")  
print(f"Best position: {pos}")
print(f"Best endurance: {endurance(pos)}")  

# Add these lines after printing the results
plt.figure(figsize=(10, 6))
plot_cost_history(optimizer.cost_history)
plt.title("PSO Convergence Plot")
plt.xlabel("Iterations")
plt.ylabel("Cost (negative endurance)")
plt.grid(True)
plt.show()

