# librerías

import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!pip install deap

from deap import base
from deap import creator
from deap import tools

# Fixed Parameters

F = 2000 # Fuerza total en lb
NDIM = 2 # Number of dimensions of the individual

LOW1, UP1 = 0, 180 # Bounds on the first gene (ángulo alfa)
LOW2, UP2 = (4/9)*math.sqrt(3), 21/8 # Bounds on the second gene (radio)
BOUNDS = [(LOW1, UP1)] + [(LOW2, UP2)]

# Hyperparameters

NGEN = 80 # Number of Generation
MU = 70 # Number of individuals in population
CXPB = 0.4 # Crossover probability
MUTPB = 0.05 #0.05 # Mutation probability
NH = 35 # Number of children
TS = 2 # Tournament Size

def init_opti():

    toolbox = base.Toolbox()
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0,-1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)# registro de operadores
    toolbox.register("individual", gen_tuple, icls = creator.Individual,
                     ranges=BOUNDS)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", evaluation)
    toolbox.register("cross", tools.cxUniform, indpb = CXPB)
    toolbox.register("mutate", mutar_ind, prob = MUTPB,
                     icls = creator.Individual, ranges=BOUNDS)
    toolbox.register("select", tools.selTournament, tournsize = TS)
    toolbox.register("delete", tools.selWorst)

# función para generar los genes
def gen_tuple(icls, ranges):
    genome = list()
    for p in BOUNDS:
        genome.append(np.random.uniform(*p))
    return icls(genome)

#función de calidad
def evaluation(ind):
    a_grados = ind[0] # ángulo en grados
    a = (a_grados*math.pi)/180 # ángulo en radianes
    r = ind[1]
    objective1 = (F/3)*math.sqrt(pow((9/r)*math.sin(a), 2)
                                 + pow(1 + (9/r)*math.cos(a), 2))
    objective2 = (F/3)*math.sqrt(pow((9/r)*math.sin(a + (2/3)*math.pi), 2)
                                 + pow(1 + (9/r)*math.cos(a + (2/3)*math.pi), 2))
    objective3 = (F/3)*math.sqrt(pow((9/r)*math.sin(a + (4/3)*math.pi), 2)
                                 + pow(1 + (9/r)*math.cos(a + (4/3)*math.pi), 2))
    return objective1, objective2, objective3

#función de mutación
def mutar_ind(ind, prob, icls, ranges):
    
    # mutación del 1er gen
    n = random.random()
    if n < prob:
        ind[0] = gen_tuple(icls, ranges)[0]
    
    # mutación del 2do gen
    n = random.random()
    if n < prob:
        ind[1] = gen_tuple(icls, ranges)[1]
    return ind

def main():

    #statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("std", np.std)
    pareto = tools.ParetoFront()
    logbook = tools.Logbook()

    pop = toolbox.population(n=MU)

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(" Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of de individuals
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < NGEN:
        g = g + 1

        # Select the next generation individuals
        offspring = toolbox.select(pop, NH)

        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for ind1, ind2 in zip(offspring[0::2], offspring[1::2]):
            # cross two individuals
            toolbox.cross(ind1, ind2)
        
        for mutant in offspring:   
            # mutate an individual
            toolbox.mutate(mutant)
            del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Eliminar individuos
        for i in range(NH):
            selected = toolbox.delete(pop, 1)[0]
            pop.remove(selected)

        # Añadir nuevos a población
        pop.extend(offspring)
        
        pareto.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, **record)
    print("-- End of (successful) evolution --")
    
    #guardar estadisticas
    gen, std = logbook.select("gen", "std")
    
    return pareto, gen, std

#grafico del pareto
def plot(data):
    
    # Se obtienen los datos
    x, y, z = zip(*[ind.fitness.values for ind in data])
    x = np.array([x])
    y = np.array([y])
    z = np.array([z])
    
    fig = plt.figure()
    fig.set_size_inches(15,10)
    
    # Pareto F2 vs F1
    axe = plt.subplot2grid((2,2),(0,0))
    axe.set_ylabel('F2 (lb)')
    axe.scatter(x, y, c='b', marker='+')# Pareto F3 vs F1
    axe = plt.subplot2grid((2,2),(1,0))
    axe.set_ylabel('F3 (lb)')
    axe.set_xlabel('F1 (lb)')
    axe.scatter(x, z, c='b', marker='+')
    
    #Pareto F3 vs F2
    axe = plt.subplot2grid((2,2),(1,1))
    axe.set_xlabel('F2 (lb)')
    scat = axe.scatter(y, z, c='b', marker='+')
    
    plt.show()
    
    # Pareto en 3D
    fig = plt.figure()
    fig.set_size_inches(8,6)
    
    axe = fig.add_subplot(111,projection='3d')
    axe.set_xlabel('F1 (lb)')
    axe.set_ylabel('F2 (lb)')
    axe.set_zlabel('F3 (lb)')
    axe.plot_wireframe(x, y, z)
    
    plt.show()
    
#grafico de la desviación estándar
def plot_stats(gen, std):
    fig, ax1 = plt.subplots()
    line = ax1.plot(gen, std, "b-", label="Standard Deviation")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Standard Deviation")
   
    labs = [l.get_label() for l in line]
    ax1.legend(line, labs, loc="center right")
    
    plt.show()

# tabulación del pareto
def tabular(pareto):
    # obtención de los datos
    F1, F2, F3 = zip(*[ind.fitness.values for ind in pareto])
    
    # listas para almacenar el contenido de la tabla
    tabla = list()
    std = list()
    
    # redondeo y ordenamiento de los datos
    for number in range(len(pareto)):
        fila = list()
        fila.append(round(pareto[number][0], 3))
        fila.append(round(pareto[number][1], 3))
        fila.append(round(F1[number], 2))
        fila.append(round(F2[number], 2))
        fila.append(round(F3[number], 2))
        desv_est = [F1[number], F2[number], F3[number]]
        fila.append(round(np.std(desv_est), 3))
        std.append(np.std(desv_est))
        tabla.append(fila)
    
    # etiquetas de la tabla
    labels = ["Ángulo (°)", "Radio (in)", "F1 (lb)", "F2 (lb)",
              "F3 (lb)", "Std"]
    
    # confección de la tabla
    df = pd.DataFrame(tabla, columns = labels)
    pd.set_option('display.max_rows', None)
    print(df)
    
    # selección del individuo con menor desviación estándar
    min_std = round(min(std), 3)
    print("El individuo con la menor desviación estándar es:")
    select = list()
    for element in tabla:
        if min_std == element[5]:
            select.append(element)
    df = pd.DataFrame(select, columns = labels)
    print(df)
    
# ejecución del programa
init_opti()
pareto, gen, std = main()
plot_stats(gen, std)
plot(pareto)
tabular(pareto)
