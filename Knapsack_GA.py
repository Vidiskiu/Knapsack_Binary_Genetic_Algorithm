import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TKAgg')

class Allele:
    def __init__(self, value:int, weight:int):
        self.value = value
        self.weight = weight

    def __str__(self):
        return "{} - {}".format(self.value, self.weight)

class Individual:
    def __init__(self, genes:list = []):
        self.chromosome = genes

    def randomize(self, geneSize:int):
        self.chromosome = np.random.randint(0, 2, geneSize)
        return self

    def __str__(self):
        return "{}".format(self.chromosome)

class GeneticAlgorithm:
    def __init__(self, alleles:list, constraint:int, populationSize:int, pc:float, pm:float):
        self.alleles = alleles
        self.constraint = constraint
        self.populationSize = populationSize
        self.population = []
        self.pc = pc
        self.pm = pm
        self.maxFitness = []
        self.avgFitness = []

        # initialize random population
        for i in range(0, populationSize):
            self.population.append(Individual().randomize(len(alleles)))

        print(end="\n" + "=" * 20 + "\n")
        print("Initialization", end="\n" + "=" * 20 + "\n")

        print("Alelles (Val, Weight)", end="\n" + "-" * 20 + "\n")
        for allele in self.alleles:
            print(allele)
        
        print("Constraint : {} | Population Size : {} | Pc : {} | Pm : {}".format(self.constraint, self.populationSize, self.pc, self.pm))

    def fitness(self, individual:Individual): # calculate fitness
        fitness = 0
        weight = 0

        for gene, allele in zip(individual.chromosome, self.alleles):
            fitness += gene * allele.value
            weight += gene * allele.weight
        else:
            if (weight > self.constraint):
                fitness -= 10
        
        return fitness

    def selectionRWS(self):
        fitness = []
        probabilities = []
        cummProbabilities = []

        print("-> Fitness Evaluation", end="\n" + "-" * 25 + "\n")

        for individual in self.population:
            individualFitness = self.fitness(individual)
            fitness.append(individualFitness)
            print("{} | Fitness : {}".format(individual, individualFitness))

        for individualFitness in fitness:
            probabilities.append(individualFitness / sum(fitness))
        
        zipped = zip(self.population, probabilities)
        self.population, probabilities = zip(*sorted(zipped, key=lambda key: key[1])) # individuals are sorted
        self.population = list(self.population)

        for i in range(0, len(probabilities)):
            totalProb = 0
            for j in range(0, i + 1):
                totalProb += probabilities[j]
            cummProbabilities.append(totalProb)

        zipped = zip(self.population, cummProbabilities)
        _, cummProbabilities = zip(*sorted(zipped, key=lambda key: key[1]))
        
        print("-> RWS", end="\n" + "-" * 10 + "\n")
        rwsRandom = sorted(np.random.rand(int(self.pc * self.populationSize)))
        print("Random Picks :", rwsRandom)

        for cummProbability, individual in zip(cummProbabilities, self.population):
            print("{} | Cummulative Probability : {}".format(individual, cummProbability))

        selectIndex = []
        for rand in rwsRandom:
            counter = 0
            for cummProbability in cummProbabilities:
                if (rand < cummProbability):
                    selectIndex.append(counter)
                    break
                counter += 1

        print("Selected Index :", selectIndex)

        return selectIndex

    def crossover(self):
        selectIndex = self.selectionRWS()
        offsprings = []
        random = np.random.randint(0, len(self.alleles))
        
        for i in range(0, int((self.pc * self.populationSize) / 2)):
            offspring = Individual(np.concatenate((self.population[selectIndex[2 * i]].chromosome[:random], self.population[selectIndex[2 * i + 1]].chromosome[random:])))
            offsprings.append(offspring)

            offspring = Individual(np.concatenate((self.population[selectIndex[2 * i + 1]].chromosome[:random], self.population[selectIndex[2 * i]].chromosome[random:])))
            offsprings.append(offspring)

        print("-> Crossover Offsprings", end="\n" + "-" * 25 + "\n")
        print("Crossover point :", random)

        for offspring in offsprings:
            print(offspring)

        return offsprings

    def mutation(self):
        offsprings = self.crossover()
        totalAlleles = int(self.pc * self.populationSize) * len(self.alleles)
        random = np.random.randint(0, totalAlleles, int(totalAlleles * self.pm))
        
        for number in random:
            offsprings[number // len(self.alleles)].chromosome[number % len(self.alleles)] = 1 if offsprings[number // len(self.alleles)].chromosome[number % len(self.alleles)] == 0 else 0

        print("-> Mutation", end="\n" + "-" * 15 + "\n")
        print("Random mutation numbers", random)
        
        for offspring in offsprings:
            print(offspring)

        return offsprings

    def elitism(self):
        self.population = [i for i in self.population]
        offsprings = self.mutation()
        self.population.extend(offsprings)
        
        fitness = []

        for individualFitness in [self.fitness(individual) for individual in self.population]:
            fitness.append(individualFitness)
        
        zipped = zip(self.population, fitness)
        self.population, fitness = zip(*sorted(zipped, key=lambda key: key[1], reverse = True)) # individuals are sorted
        self.population = list(self.population)

        print("-> Population before elitism", end="\n" + "-" * 15 + "\n")
        for individual in self.population:
            individualFitness = self.fitness(individual)
            print("{} | Fitness : {}".format(individual, individualFitness))

        self.population = self.population[: self.populationSize]
        fitness = fitness[: self.populationSize]
        
        self.maxFitness.append(max(fitness))
        self.avgFitness.append(sum(fitness)/len(fitness))
        
    def evolve(self, generation: int):
        print(end="\n" + "=" * 20 + "\n")
        print("Evolving", end="\n" + "=" * 20 + "\n")

        for counter in range(generation):
            print("=> Generation", counter + 1)
            self.elitism()
        
        fitness = []
        for individual in self.population:
            individualFitness = self.fitness(individual)
            fitness.append(individualFitness)

        print("| End | Best solution is {} with fitness {}".format(self.population[0], max(fitness)))

    def graph(self):
        x = np.linspace(1, len(self.avgFitness), len(self.avgFitness))
        
        plt.plot(x, self.avgFitness, label = "Average")
        plt.plot(x, self.maxFitness, label = "Max")
        plt.show()

"""
Manual input
-------------
n = int(input("Enter numbers of items : "))
items = []

for i in range(0, n):
    print("Item", i+1)
    value = int(input("Value : "))
    weight = int(input("Weight : "))
    item = Item(value, weight)
    items.append(item)
"""

def __main__():
    # Variable initialization
    values = [5, 8, 3, 2, 7, 9, 4]
    weights = [7, 8, 4, 10, 4, 6, 4]
    alleles = []
    constraint = 30
    generation = 50
    populationSize = 10
    pc = 0.8
    pm = 0.1

    for i in range(0, len(values)): # create items list
        value = values[i]
        weight = weights[i]
        allele = Allele(value, weight)
        alleles.append(allele)

    t = time.time()
    GA = GeneticAlgorithm(alleles, constraint, populationSize, pc, pm)  # initialize GA class
    GA.evolve(generation)
    print("time elapsed : {} s".format(time.time() - t))
    GA.graph()

if __name__ == "__main__":
    __main__()