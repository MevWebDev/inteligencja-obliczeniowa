import matplotlib.pyplot as plt
import random
import time
import numpy as np

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
)

COORDS_LARGE = (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90),
    (87, 83), (73, 23), (40, 35), (10, 10), (50, 90),
    (60, 30), (25, 70), (95, 45), (15, 45), (80, 10),
    (35, 30), (55, 72), (90, 20), (65, 95), (75, 55)
)

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS_LARGE, ant_count=300, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                    iterations=3)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()



def test_parameters(coords, param_sets):
    results = []
    
    for params in param_sets:
        start_time = time.time()
        
        colony = AntColony(
            coords, 
            ant_count=params['ant_count'], 
            alpha=params['alpha'], 
            beta=params['beta'],
            pheromone_evaporation_rate=params['evaporation'], 
            pheromone_constant=params['pheromone_constant'],
            iterations=params['iterations']
        )
        
        path = colony.get_path()
        
        # Oblicz długość ścieżki
        path_length = 0
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            path_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Dodaj odległość od ostatniego do pierwszego (pełny cykl)
        x1, y1 = path[-1]
        x2, y2 = path[0]
        path_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        execution_time = time.time() - start_time
        
        results.append({
            'params': params,
            'path_length': path_length,
            'execution_time': execution_time
        })
        
        print(f"Parametry: {params}")
        print(f"  Długość ścieżki: {path_length:.2f}")
        print(f"  Czas wykonania: {execution_time:.2f} s")
    
    return results

# Definicje zestawów parametrów do testów
parameter_sets = [
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 50},
    {'ant_count': 300, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 50},
    {'ant_count': 100, 'alpha': 1.0, 'beta': 1.0, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 50},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 2.5, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 50},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.2, 'pheromone_constant': 1000, 'iterations': 50},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.7, 'pheromone_constant': 1000, 'iterations': 50}
]

# Uruchom testy
print("Testowanie dla małego zbioru danych:")
results_small = test_parameters(COORDS, parameter_sets)

# print("\nTestowanie dla dużego zbioru danych:")
# results_large = test_parameters(COORDS_LARGE, parameter_sets)

# Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(
    [f"Set {i+1}" for i in range(len(results_small))], 
    [r['path_length'] for r in results_small]
)
plt.title('Długość ścieżki - Mały zbiór')
plt.ylabel('Długość ścieżki')
plt.xticks(rotation=45)
plt.savefig('small_dataset_results.png', dpi=300)

plt.subplot(1, 2, 2)
plt.bar(
    [f"Set {i+1}" for i in range(len(results_small))], 
    [r['execution_time'] for r in results_small]
)
plt.title('Czas wykonania - Mały zbiór')
plt.ylabel('Czas (s)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Modyfikacja kodu dla algorytmu ACO - Zadanie 2

## b) Generowanie większej liczby wierzchołków


# Wersja z większą liczbą wierzchołków (20 miast)
COORDS_LARGE = (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90),
    (87, 83), (73, 23), (40, 35), (10, 10), (50, 90),
    (60, 30), (25, 70), (95, 45), (15, 45), (80, 10),
    (35, 30), (55, 72), (90, 20), (65, 95), (75, 55)
)

# Użyj tego kodu, aby przetestować większą liczbę wierzchołków
plot_nodes()
colony = AntColony(COORDS_LARGE, ant_count=300, alpha=0.5, beta=1.2, 
                pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                iterations=300)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )
plt.plot(
    (optimal_nodes[-1][0], optimal_nodes[0][0]),
    (optimal_nodes[-1][1], optimal_nodes[0][1]),
)  # Połącz ostatni z pierwszym
plt.show()


## c) Eksperymenty z parametrami


import time
import numpy as np

def test_parameters(coords, param_sets):
    results = []
    
    for params in param_sets:
        start_time = time.time()
        
        colony = AntColony(
            coords, 
            ant_count=params['ant_count'], 
            alpha=params['alpha'], 
            beta=params['beta'],
            pheromone_evaporation_rate=params['evaporation'], 
            pheromone_constant=params['pheromone_constant'],
            iterations=params['iterations']
        )
        
        path = colony.get_path()
        
        # Oblicz długość ścieżki
        path_length = 0
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            path_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Dodaj odległość od ostatniego do pierwszego (pełny cykl)
        x1, y1 = path[-1]
        x2, y2 = path[0]
        path_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        execution_time = time.time() - start_time
        
        results.append({
            'params': params,
            'path_length': path_length,
            'execution_time': execution_time
        })
        
        print(f"Parametry: {params}")
        print(f"  Długość ścieżki: {path_length:.2f}")
        print(f"  Czas wykonania: {execution_time:.2f} s")
    
    return results

# Definicje zestawów parametrów do testów
parameter_sets = [
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 100},
    {'ant_count': 300, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 100},
    {'ant_count': 100, 'alpha': 1.0, 'beta': 1.0, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 100},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 2.5, 'evaporation': 0.4, 'pheromone_constant': 1000, 'iterations': 100},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.2, 'pheromone_constant': 1000, 'iterations': 100},
    {'ant_count': 100, 'alpha': 0.5, 'beta': 1.2, 'evaporation': 0.7, 'pheromone_constant': 1000, 'iterations': 100}
]

# Uruchom testy
print("Testowanie dla małego zbioru danych:")
results_small = test_parameters(COORDS, parameter_sets)

print("\nTestowanie dla dużego zbioru danych:")
results_large = test_parameters(COORDS_LARGE, parameter_sets)

# Wizualizacja wyników
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(
    [f"Set {i+1}" for i in range(len(results_small))], 
    [r['path_length'] for r in results_small]
)
plt.title('Długość ścieżki - Mały zbiór')
plt.ylabel('Długość ścieżki')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(
    [f"Set {i+1}" for i in range(len(results_small))], 
    [r['execution_time'] for r in results_small]
)
plt.title('Czas wykonania - Mały zbiór')
plt.ylabel('Czas (s)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


### Wnioski, które warto uwzględnić w analizie:

# 1. **Wpływ liczby mrówek (ant_count)**:
#    - Większa liczba mrówek zwiększa szansę znalezienia dobrego rozwiązania
#    - Ale zwiększa też czas obliczeń

# 2. **Wpływ parametrów alpha i beta**:
#    - Alpha (α) kontroluje wpływ feromonów - wyższe wartości powodują, że mrówki bardziej podążają za feromonami
#    - Beta (β) kontroluje wpływ odległości - wyższe wartości powodują, że mrówki bardziej preferują krótsze drogi
#    - Dobry balans jest kluczowy: α=0.5-1.0, β=1.0-2.5

# 3. **Wpływ współczynnika parowania feromonów**:
#    - Niski (0.1-0.3): powolna zmiana, stabilność, ale ryzyko utknięcia w lokalnym optimum
#    - Wysoki (0.6-0.9): szybka adaptacja, więcej eksploracji, ale może nie wykształcić wyraźnej ścieżki

# 4. **Stała feromonowa**: Wpływa na bezwzględną ilość odkładanego feromonu

