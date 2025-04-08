import pygad
import numpy

import numpy as np
import matplotlib.pyplot as plt

# Define the maze as a matrix: 0 = path, 1 = wall
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
])

# Define start and end positions (row, column)
start_pos = (1, 1)  # Top-left corner
end_pos = (10, 10)    # Bottom-right corner

# Function to display the maze
def display_maze(maze, path=None):
    plt.figure(figsize=(12, 12))
    
    # Create a colormap: 0=white (path), 1=blue (wall)
    colored_maze = np.zeros((maze.shape[0], maze.shape[1], 3))
    
    # Set walls as blue
    colored_maze[maze == 0] = [0, 0, 1]  # Blue for walls
    
    # Set path as white
    colored_maze[maze == 1] = [1, 1, 1]  # White for paths
    
    # If a path is provided, mark it
    if path is not None:
        for pos in path:
            colored_maze[pos] = [0, 1, 0]  # Green for the solution path
    
    # Mark start position as red
    colored_maze[start_pos] = [1, 0, 0]  # Red
    
    # Mark end position as purple
    colored_maze[end_pos] = [1, 0, 1]  # Purple
    
    plt.imshow(colored_maze)
    
    # Add grid
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks(np.arange(-.5, maze.shape[1], 1), [])
    plt.yticks(np.arange(-.5, maze.shape[0], 1), [])
    
    plt.title('Maze with Start (Red) and End (Purple)')
    plt.show()



#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1, 2, 3] # góra prawo dół lewo
num_genes = 50

#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx):
    current_pos = start_pos
    path = [current_pos]
    
    
    # Keep track of visited positions
    visited = set([current_pos])
    
    reached_end = False
    steps_taken = 0
    backtrack_count = 0  # Count how many times we revisit positions
    wall_hits = 0

    for move in solution:
        new_row, new_col = current_pos

        if move == 0:
            new_row -= 1
        elif move == 1:  # Right
            new_col += 1
        elif move == 2:  # Down
            new_row += 1
        elif move == 3:  # Left
            new_col -= 1
            
        steps_taken += 1
        
        # Calculate potential new position
        new_pos = (new_row, new_col)
        
        # Check if it's a valid move (within bounds and on a path)
        if (0 <= new_row < maze.shape[0] and 
            0 <= new_col < maze.shape[1] and 
            maze[new_row, new_col] == 1):
            
            # Valid move, update position
            current_pos = new_pos
            
            # Check if we're revisiting a position
            if new_pos in visited:
                backtrack_count += 1
            else:
                visited.add(new_pos)
                
            path.append(current_pos)
            
            # Check if we reached the end
            if current_pos == end_pos:
                reached_end = True
                break
        else:
            wall_hits += 1 # Hit a wall or went out of bounds, stay in place
            pass
    
    # Calculate base fitness from distance to end
    distance_to_end = abs(current_pos[0] - end_pos[0]) + abs(current_pos[1] - end_pos[1])
    fitness = 1.0 / (1.0 + distance_to_end)
    
    # Apply penalties and bonuses
    if reached_end:
        # Large bonus for reaching the end
        fitness += 100
        # Bonus for shorter paths
        fitness += 10 * (num_genes - steps_taken)
    
    # Heavy penalty for backtracking (revisiting positions)
    fitness -= 5 * backtrack_count

    fitness -= 2* wall_hits
    
    return fitness, path, wall_hits, backtrack_count

def fitness_wrapper(model, solution, solution_idx):
    fitness_value, _, _, _ = fitness_func(model, solution, solution_idx)
    return fitness_value

fitness_function = fitness_wrapper

sol_per_pop = 200  # Larger population for better exploration
num_parents_mating = 20
num_generations = 400  # More generations to find solution
parent_selection_type = "tournament"
keep_parents = 10
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 15

def runAlgorythm():
    #inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
    ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution:", solution_fitness)

    # Visualize the best path
    best_fitness, best_path, wall_hits, backtrack_count = fitness_func(None, solution, 0)
    print(f"Path length: {len(best_path)}")
    print(f"Reached end: {best_path[-1] == end_pos}")
    print(f"Wall hits: {wall_hits}")
    print(f"Backtracking count: {backtrack_count}")

    return best_path, wall_hits, backtrack_count


shortest_path = None
shortest_path_length = float('inf')
best_wall_hits = float('inf')
best_backtrack_count = float('inf')

for i in range(10):
    print(f"\nRun {i+1}/10:")
    path, wall_hits, backtrack_count = runAlgorythm()
    
    # Check if this path reaches the end
    if path[-1] == end_pos:
        # If this is our first valid path or it's shorter than what we have
        if shortest_path is None or len(path) < len(shortest_path):
            shortest_path = path
            shortest_path_length = len(path)
            best_wall_hits = wall_hits
            best_backtrack_count = backtrack_count
            print(f"New shortest path found! Length: {shortest_path_length}, Wall hits: {wall_hits}, Backtracking: {backtrack_count}")

# After all runs, display the best result
if shortest_path:
    print("\n=== FINAL RESULTS ===")
    print(f"Shortest successful path length: {len(shortest_path)}")
    print(f"Path reached end: {shortest_path[-1] == end_pos}")
    print(shortest_path)
    
    # Display the maze with the shortest path
    display_maze(maze, shortest_path)
    
    # Plot the generations graph for the best run
    # Note: We'd need to modify runAlgorythm() to return the GA instance as well
else:
    print("No successful path to the end was found in any run.")




