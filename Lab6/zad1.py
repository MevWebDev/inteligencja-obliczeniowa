import pygad
import numpy
import time

item_weights = [7,7,6,2,5,6,1,3,10,3,15] 
item_values = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]  
max_capacity = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx):
    total_weight = numpy.sum(solution * item_weights)
    total_value = numpy.sum(solution * item_values)
    if total_weight > max_capacity:
        return -1
    return total_value

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(item_values)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

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

start = time.time()
ga_instance.run()
end = time.time()
time_elapsed = end - start

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# Calculate and display details of the selected items
selected_weight = numpy.sum(item_weights * solution)
selected_value = numpy.sum(item_values * solution)
print(f"Total weight of selected items: {selected_weight}/{max_capacity}")
print(f"Total value of selected items: {selected_value}")

# List the selected items
print("Selected items (index, weight, value):")
backpack_value = 0
for i in range(len(solution)):
    if solution[i] == 1:
        backpack_value += item_values[i]
        print(f"Item {i}: Weight={item_weights[i]}, Value={item_values[i]}")
print(f"Backpack value: {backpack_value}")
print(f"Algorythm finished in {time_elapsed}")
# Display the fitness progress
ga_instance.plot_fitness()


overall_time = 0
overall_values = 0
for i in range(10):
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
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria="reach_1630"
                       )
    
    start = time.time()
    ga_instance.run()
    end = time.time()
    overall_time += (end - start)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Total value of selected items: {selected_value}")
    overall_values += selected_value

print(f"Average time of 10 attemps is: {overall_time / 10}")
print(f"Average value of backpack for 10 attemps is: {overall_values / 10}")
