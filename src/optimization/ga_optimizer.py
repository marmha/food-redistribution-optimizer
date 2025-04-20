import random
import numpy as np
from deap import base, creator, tools, algorithms

# Example distance matrix (distance[i][j] = distance from location i to j)
distance_matrix = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

NUM_LOCATIONS = len(distance_matrix)

# 1. Define fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize distance
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 2. Gene = delivery location index (excluding the starting point 0)
toolbox.register("indices", random.sample, range(1, NUM_LOCATIONS), NUM_LOCATIONS - 1)

# 3. Structure initial individual and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 4. Evaluation function = total route distance (starting and returning to 0)
def evalRoute(individual):
    route = [0] + individual + [0]  # Start and end at 0 (e.g., warehouse)
    total_distance = sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    return (total_distance,)

toolbox.register("evaluate", evalRoute)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(pop_size=100, generations=100):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                   ngen=generations, stats=stats, halloffame=hof, verbose=True)

    best_route = [0] + hof[0] + [0]
    print("\n🚚 Best Route Found:", best_route)
    print("🛣️ Total Distance:", evalRoute(hof[0])[0])

    return best_route, evalRoute(hof[0])[0]

if __name__ == "__main__":
    run_ga()
