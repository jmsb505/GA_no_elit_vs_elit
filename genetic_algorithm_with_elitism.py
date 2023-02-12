from deap import base
from deap import creator
from deap import tools
from geopy import distance
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results = {}
BEST_RESULTS = []
random.seed(42)
hof=tools.HallOfFame(30)

city_mapping = {
    0: ["seattle", (47.608013, -122.335167)],
    1: ["boise", (43.616616, -116.200886)],
    2: ["everett", (47.967306, -122.201399)],
    3: ["pendleton", (45.672075, -118.788597)],
    4: ["biggs", (45.669846, -120.832841)],
    5: ["portland", (45.520247, -122.674194)],
    6: ["twin_falls", (42.570446, -114.460255)],
    7: ["bend", (44.058173, -121.315310)],
    8: ["spokane", (47.657193, -117.423510)],
    9: ["grant_pass", (42.441561, -123.339336)],
    10: ["burns", (43.586126, -119.054413)],
    11: ["eugene", (44.050505, -123.095051)],
    12: ["lakeview", (42.188772, -120.345792)],
    13: ["missoula", (46.870105, -113.995267)]
}

# problem constants:
PATH_LENGTH = 13  # length of bit string to be optimized
PATH = list()

# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 200

# set the random seed:

toolbox = base.Toolbox()

def get_distance(city1, city2):
    return distance.distance(city_mapping[city1][1], city_mapping[city2][1]).km

def print_individual(individual):
    assert isinstance(individual, list)
    path = f'{city_mapping[0][0]} -> '

    for city in individual:
        if city == 0: city = 13
        path += f'{city_mapping[city][0]} -> '

    path += f'{city_mapping[0][0]}'

    print(path)


def get_city():
    city = random.randint(0, 12)

    while city in PATH:
        city = random.randint(0, 12)

    PATH.append(city)

    if len(PATH) == 13:
        PATH.clear()

    return city

def minimum_distance(individual):
    start = 0
    path_distance = 0

    for city in individual:
        if city == 0:
            city = 13
        path_distance += get_distance(start, city)
        start = city

    return (path_distance + get_distance(start, 0)),

def plot_linear_graph(individual, title):
    x_values = [city_mapping[city][1][0] for city in individual]
    y_values = [city_mapping[city][1][1] for city in individual]
    plt.plot(x_values, y_values, color='blue', marker='o')
    plt.title(title, fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    
    for x, y, city in zip(x_values, y_values, individual):
        label = city
        plt.annotate(label, (x, y),
                    xycoords="data",
                    textcoords="offset points",
                    xytext=(0, 10), ha="center")
    plt.show()

toolbox.register("create_path", get_city)

creator.create("minimumFitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.minimumFitness)

toolbox.register("indexes", random.sample, range(13), 13)
toolbox.register("createIndividual", tools.initIterate, creator.Individual,
                 toolbox.indexes)

#toolbox.register("createIndividual", tools.initRepeat, creator.Individual, toolbox.indexes, 13)
toolbox.register("createPopulation", tools.initRepeat, list, toolbox.createIndividual)
toolbox.register("evaluate", minimum_distance)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/PATH_LENGTH)

def main():
    population = toolbox.createPopulation(n=POPULATION_SIZE)
    generationCounter = 0

    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []
    best_final_individual = []
    
    best_first_individual = list.copy(population[fitnessValues.index(min(fitnessValues))])
    for x in range(len(best_first_individual)):
        best_first_individual[x]+=1
    best_first_individual.insert(0,0)
    best_first_individual.append(0)
    plot_linear_graph(best_first_individual, 'Best First Individual Before Generations')
    
    while generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1
        hof.update(population)
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        hofl=list(map(toolbox.clone, hof))
        
        for i in hofl:
            ln=len(offspring)
            for x in range(ln):
                if(np.array_equal(i,offspring[x])):
                    offspring.pop(x)
                    break

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))

        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = hofl + offspring
        population=population[:300]
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = min(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationCounter, maxFitness, meanFitness))

        # find and print best individual:
        best_index = fitnessValues.index(min(fitnessValues))
        best_final_individual = list.copy(population[best_index])
        print("Best Individual = ", *population[best_index], "\n")
        print_individual(population[best_index])
        results[maxFitness] = population[best_index]
        BEST_RESULTS.append(maxFitness)
        BEST_RESULTS.sort()

    for x in range(len(best_final_individual)):
        best_final_individual[x]+=1
    best_final_individual.insert(0,0)
    best_final_individual.append(0)

    plot_linear_graph(best_final_individual, 'Best Individual From All Generations')
    # Genetic Algorithm is done - plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()








