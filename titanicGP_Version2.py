# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:28:32 2018

@author: User
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from deap import creator, tools, base, gp, algorithms
from sklearn.metrics import confusion_matrix
import operator

def preprocessData():
    #Importing the dataset
    dataset = pd.read_csv('train.csv')
    X = dataset[['Pclass','Age', 'Sex','SibSp','Parch','Fare']].values
    Y = dataset.iloc[:, 1].values

    #Dealing with encoding the labels
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,2] = labelencoder_X.fit_transform(X[:,2])

    #Dealing with missing datas
    from sklearn.preprocessing import Imputer
    imputer = Imputer()
    imputer=Imputer(missing_values='NaN',
                    strategy='mean', axis=0)
    imputer=imputer.fit(X[:,1:2])
    X[:,1:2]=imputer.transform(X[:,1:2])
    return X, Y

#Number of attributes
ATTRIBUTE_NUM = 6

#Creating the individuals
#lists of floats with a minimizing objectives fitness
creator.create("FitnessMin", base.Fitness, weights = (-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMin)

#initialize a primitive set and add all the primitives our trees can use
pset = gp.PrimitiveSet("MAIN", ATTRIBUTE_NUM, "IN")
pset.addPrimitive(np.add, arity=2)
pset.addPrimitive(np.subtract, arity=2)
pset.addPrimitive(np.multiply, arity=2)
pset.addPrimitive(np.negative, arity=1)
#Addional primitives? ========================================================
pset.addPrimitive(np.absolute, arity=1)
pset.addPrimitive(np.log, arity=1)
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')
pset.addTerminal(0)
pset.addTerminal(1)

toolbox = base.Toolbox()
#Returns a primitive tree based on a primitive set and a minimum and maximum tree depth
#Maybe change the size of the tree?
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

inputs, outputs = preprocessData()

def evaluateInd(individual, pset):
    func = gp.compile(expr=individual, pset=pset)
    predictions = [(func(*in_)>0) for in_ in inputs]
    tn, fp, fn, tp = confusion_matrix(predictions, outputs).ravel()
    return fp,fn

def movingAverage(values, window):
    weights1 = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights1, 'valid')
    return sma

toolbox.register("evaluate", evaluateInd, pset = pset)

'''
CHANGE THE THINGS HERE!!!!!!
==============================================================================
'''
#Change the params here for mating, mutating, and selecting
toolbox.register("mate", gp.cxOnePoint)
#there is a variety of mutation operators in the deap.tools module
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
#https://github.com/DEAP/deap/blob/master/deap/gp.py <-- more mutator methods
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

#Optional, limits tree size
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
'''
==============================================================================
'''

def main():

    NGEN = 50
    MU = 50
    LAMBDA = 100
    CXPB = 0.5
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    gen = range(NGEN)
    avg_list = []
    max_list = []
    min_list = []
    AUC = []

    #======================================================
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for g in gen:
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            print(ind)
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
#you did some shit here rohti
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        g_max = max(fits)
        g_min = min(fits)

        avg_list.append(mean)
        max_list.append(g_max)
        min_list.append(g_min)

        print("  Min %s" % g_min)
        print("  Max %s" % g_max)
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        #THIS IS VERSION 2 ===========================================================================
        """Split fitness values into separate lists"""
        fitness_1 = [ind.fitness.values[0] for ind in pop]
        fitness_2 = [ind.fitness.values[1] for ind in pop]

        f1 = np.array(fitness_1)
        f2 = np.array(fitness_2)

        """Calculate area under curve with least squares method"""
        print("Area Under Curve: %s" % (np.sum(np.abs(np.diff(f1))*f2[:-1])))
        AUC.append((np.sum(np.abs(np.diff(f1))*f2[:-1])))
        #=============================================================================================

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness)) we dont need this
    # gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    # plt.plot(gen, avg, label="average")
    # plt.plot(gen, min_, label="minimum")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.legend(loc="upper left")
    # plt.show()

    plt.subplot(1, 2, 1)
    plt.plot(gen, avg_list, label="average")
    plt.plot(gen, min_list, label="minimum")
    plt.plot(gen, max_list, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(gen, AUC, label="AUC")
    plt.xlabel("Generation")
    plt.ylabel("AUC")

    plt.show()

main()










