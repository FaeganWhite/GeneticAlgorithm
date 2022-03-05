# Imports
import random

# A class for an individual solution
class individual:
    # Initiate the self
    def __init__(self, n):
        # Create some empty genes
        self.gene = [0]*n
        # Create a variable to store the fitness of te individual solution
        self.fitness = 0

# Finess function, a simple sum of all the values in the gene
def findFitness(gene):
    # Create a fitness variable, initilaising it as 0
    fitness = 0
    # For every value in the gene
    for i in gene:
        # add the value to the total fitness 
        fitness += i
    # Return the fitness of the solution
    return fitness

# Function to generate a random gene
def generateRandomGene(n):
    # Create an empty array
    gene = []
    # For n times
    for x in range(n):
        # Generate a random binary value
        random_value = round(random.random()*10)%2
        # Add a value to the gene
        gene.append(random_value)
    # Return the gene
    return gene

# Initiate and return a population of random individuals
def initiatePopulation(size, genes):
    # Create a population array
    population = []
    # For the given size
    for x in range(size):
        # Create a new individual
        new_indiviual = individual(genes)
        # Set its genes to random
        new_indiviual.gene = generateRandomGene(genes)
        # Add the individual to the population
        population.append(new_indiviual)
    # Return the population
    return population

# Take a population and get a set of offspring
def evaluateOffspring(population, number_of_parents):
    # For every individual
    for individual in population:
        # Find its fitness
        individual.fitness = findFitness(individual.gene)
    # Create an offspring array
    offspring = []
    # For every member of the population
    for x in range(len(population)):
        # Get a random selection of parents
        parents = []
        # For the number of parents
        for y in range(number_of_parents):
            # Add a random parent to the list of parents
            parents.append(population[random.randint(0, len(population)-1)])
        # Choose the first parent as the fittest
        fittest_parent = parents[0]
        # For every parent
        for parent in parents:
            # if the fitness is greater than the fittest parents fitness
            if parent.fitness >= fittest_parent.fitness:
                # Make the fittest parent the new parent
                fittest_parent = parent
        # Add the
        offspring.append(fittest_parent)
    # return the offspring array
    return offspring 

# Get the average population fitness
def averagePopulationFitness(population):
    # Initialise the total fitness variable
    total_fitness = 0
    # For every individual
    for individual in population:
        # Add the fitness to the total fitness
        total_fitness += individual.fitness
    # Return a mean average of the population's fitness
    return total_fitness/len(population)

# Evaluate the population fitness
def geneticSolve(population, iterations, number_of_parents):
    # Print out the population's fitness
    print(averagePopulationFitness(population))
    #
    printPopulation(population)
    # For every iteration
    for a in range(iterations):
        print(a)
        # Evaluate the population and produce a new set of offspring
        population = evaluateOffspring(population, number_of_parents)
        # Print out the population's fitness
        print(averagePopulationFitness(population))
        #
        printPopulation(population)

# Print out the population's genes
def printPopulation(population):
    # Print a blank line
    print("")
    # For each individual
    for individual in population:
        # print the gene
        print(individual.gene)
        # Print a blank line
        print("")
    # Print a blank line
    print("")
    # Print a blank line
    print("")
    

# Initiate the population
population = initiatePopulation(5, 10)
# Run the genetic algorithm
geneticSolve(population, 5, 2)
