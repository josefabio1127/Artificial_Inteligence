#Elaborado por:
#César Argüello Salas
#Jose Fabio Navarro Naranjo

#instalación de la librería DEAP
!pip install deap

#importación de librerías adicionales
import random
import numpy
import math
import matplotlib.pyplot as plt

#importación de módulos de DEAP
from deap import base
from deap import creator
from deap import tools

#se define que el problema es de minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

#creación de la clase para los individuos
creator.create("Individual", list, fitness=creator.FitnessMin)

alelosRC = [10, 12, 22, 33, 47, 56, 68, 82] #espacio de alelos de la mantisa

def gen_tuplaRC(): #función para generar las tuplas de genes de cada componente
    return (random.choice(alelosRC), random.randint(0, 9))

toolbox = base.Toolbox() #se inicializa el toolbox
toolbox.register("tuplaRC", gen_tuplaRC) #se registra la función para crear genes
toolbox.register("individual", tools.initRepeat, creator.Individual,
toolbox.tuplaRC, n=7) #se registra la estructura del cromosoma
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #se registra la población

def fcalidad(ind): #se define la función de calidad
    #cálculo de los valores de los componentes a partir de sus genes
    R1 = ind[0][0] * pow(10, ind[0][1]-2)
    R2 = ind[1][0] * pow(10, ind[1][1]-2)
    Rf = ind[2][0] * pow(10, ind[2][1]-2)
    Ra = ind[3][0] * pow(10, ind[3][1]-2)
    Rb = ind[4][0] * pow(10, ind[4][1]-2)
    C1 = ind[5][0] * pow(10, ind[5][1]-12)
    C2 = ind[6][0] * pow(10, ind[6][1]-12)

    #porcentaje de error de la frecuencia central
    error_f0 = abs(((1/(20000*math.pi))*math.sqrt((R1+Rf)/(R1*R2*Rf*C1*C2)))-1)

    #porcentaje de error de la calidad Q del filtro
    error_Q = abs((0.25*(math.sqrt((R1+Rf)*R1*R2*Rf*C1*C2))/((R1*Rf*(C1+C2))+(R2*C2*(Rf-((Rb/Ra)*R1)))))-1)
    
    calidad = 10*error_f0 + error_Q #suma ponderada de los errores
    
    return calidad,

def mutar_ind(ind, prob): #función para mutar individuos
    #la probabilidad de mutación le ingresa como parámetro
    n = random.random() #número aleatorio entre 0 y 1
    if n < prob: #condición para que ocurra la mutación
    idx = random.randrange(7) #se selecciona una tupla de genes al azar
    ind[idx] = gen_tuplaRC() #se modifica la tupla seleccionada
    return ind

#se registran los operadores en el toolbox
toolbox.register("select", tools.selTournament, tournsize=2) #operador de selección

#para recombinación
toolbox.register("mate", tools.cxOnePoint) #operador de recombinación
toolbox.register("mutate", mutar_ind, prob=0.1) #operador de mutación con

#probabilidad del 10%toolbox.register("evaluate", fcalidad) #función de calidad
toolbox.register("delete", tools.selWorst) #operador de selección para reemplazo

def main(): #función principal, donde se programa el algoritmo

    #definición de hiperparámetros
    SIZE_POP = 400 #Tamaño de la población
    N_HIJOS = 100 #Número de hijos por generación
    N_GENERACIONES = 200 #Número máximo de generaciones
    pop = toolbox.population(n=SIZE_POP) #se inicializa la población inicial

    #se inicializa la herramienta para generar las estadísticas
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", numpy.min) #calidad del mejor individuo
    stats.register("std", numpy.std) #desviación estándar de calidades
    
    hof = tools.HallOfFame(1) #salón de la fama para guardar al mejor individuo
    logbook = tools.Logbook() #variable que lleva registro de las estadísticas

    #Se evalúan las calidades de toda la población
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    #Se crea una lista para que contiene las calidades
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0 #variable para llevar la cuenta del número de generaciones

    while g < N_GENERACIONES: #ciclo para llevar a cabo la evolución
        g = g + 1 #se empieza una nueva generación
        
        #Selección de los individuos para recombinación (padres)
        offspring = toolbox.select(pop, N_HIJOS)
       
        #Se crea una copia de los padres
        offspring = list(map(toolbox.clone, offspring))
        
        #Se aplica el operador de recombinación para generar los hijos
        for child1, child2 in zip(offspring[0::2], offspring[1::2]):
            toolbox.mate(child1, child2)
        
        for child in offspring:
            toolbox.mutate(child)   #se aplica el operador de mutación
                                    #sobre los hijos recientemente creados
            del child.fitness.values    #se borran sus valores de calidad
                                        #porque necesitan ser reevaluados
        
        #Se evalúan las calidades de los nuevos individuos
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        
        #Se eliminan N_HIJOS de la población inicial
        #usando el operador de selección para reemplazo
        for i in range(N_HIJOS):
            selected = toolbox.delete(pop, 1)[0]
            pop.remove(selected)
        
        #Se agregan los nuevos individuos a la población
        pop.extend(offspring)
        
        #Se calculan y guardan las estadísticas de la nueva generación
        record = stats.compile(pop)
        logbook.record(gen=g, **record)
        hof.update(pop) #se actualiza el salón de la fama

    #se leen las estadísticas guardadas durante la evolución
    gen, min_fit, std_fit = logbook.select("gen", "min", "std")

    #creación del gráfico para mostrar las curvas de calidad y desv. estándar
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, min_fit, "b-", label="Calidad del mejor individuo")ax1.set_xlabel("Generación")
    ax1.set_ylabel("Calidad", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, std_fit, "r-", label="Desviación estándar de calidad")
    ax2.set_ylabel("Desviación estándar", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right")
    
    plt.show() #se muestra el gráfico
    
    #se extrae el genoma del mejor individuo y su valor de calidad
    mejor_ind = hof[0]
    calidad_ind = mejor_ind.fitness.values[0]
    
    #se calcula el valor de los componentes a partir de sus genes
    R1 = mejor_ind[0][0] * pow(10, mejor_ind[0][1]-2)
    R2 = mejor_ind[1][0] * pow(10, mejor_ind[1][1]-2)
    Rf = mejor_ind[2][0] * pow(10, mejor_ind[2][1]-2)
    Ra = mejor_ind[3][0] * pow(10, mejor_ind[3][1]-2)
    Rb = mejor_ind[4][0] * pow(10, mejor_ind[4][1]-2)
    C1 = mejor_ind[5][0] * pow(10, mejor_ind[5][1]-12)
    C2 = mejor_ind[6][0] * pow(10, mejor_ind[6][1]-12)
    
    #se calcula la frecuencia central y la calidad del filtro resultante
    f0 = (1/(2000*math.pi))*math.sqrt((R1+Rf)/(R1*R2*Rf*C1*C2)) #en KHz
    Q = (math.sqrt((R1+Rf)*R1*R2*Rf*C1*C2))/((R1*Rf*(C1+C2))+(R2*C2*(Rf-((Rb/Ra)*R1))))
    
    #se imprimen los resultados obtenidos
    print("---------------------------------------------------------------")
    print("Genoma del mejor individuo:")
    print("%s" % mejor_ind)
    print("\nCalidad del mejor individuo: %f" % calidad_ind)
    print("---------------------------------------------------------------")
    print("Valores de los componentes del filtro:")
    print("R1: %E [Ω]" % R1)
    print("R2: %E [Ω]" % R2)print("Rf: %E [Ω]" % Rf)
    print("Ra: %E [Ω]" % Ra)
    print("Rb: %E [Ω]" % Rb)
    print("C1: %E [F]" % C1)
    print("C2: %E [F]" % C2)
    print("\nFrecuencia central del filtro: %f [kHz]" % f0)
    print("\nCalidad Q del filtro: %f" % Q)
    print("---------------------------------------------------------------")


main() #llamada a la función main para ejecutar el algoritmo
