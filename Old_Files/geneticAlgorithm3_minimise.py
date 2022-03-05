# Imports
import random
import matplotlib.pyplot as plt

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

# Update the population's fitness
def updatePopulationFitness(population):
    # For every individual
    for individual in population:
        # Find its fitness
        individual.fitness = findFitness(individual.gene)

# Function to generate a random gene
def generateRandomGene(n):
    # Create an empty array
    gene = []
    # For n times
    for x in range(n):
        # Generate a random binary value
        random_value = random.random()
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
    # Update the population fitness
    updatePopulationFitness(population)
    # Return the population
    return population

# Create a child by crossing over 2 parents
def crossover(parent1, parent2):
    # Get the gene number
    size = len(parent1.gene)
    # Create an offspring array
    offspring = individual(size)
    # Define a crossover point in the length of the first parent
    crossover_point = round(random.random()*size)
    # Combine parent1 before the crossover point with parent2 after the crossover point
    offspring.gene = parent1.gene[0:crossover_point+1] + parent2.gene[crossover_point+1:len(parent1.gene)+1]
    # Return the offspring
    return offspring

# Create a child by crossing over 2 parents
def mutate(individual, mutation_chance, mutation_size):
    # For every value in the gene
    for x in range(len(individual.gene)):
        # If the random value is less than the mutation_chance
        if (random.random() < mutation_chance):
            # With 50% chance
            if round(random.random()) == 1:
                # Add a random value to the gene defined by the mutation size
                individual.gene[x] += random.randrange(round(mutation_size*1000))/1000
            # otherwise
            else:
                # Subtract a random value from the gene defined by the mutation size
                individual.gene[x] -= random.randrange(round(mutation_size*1000))/1000
        # If the individual gene is greater than 1
        if individual.gene[x] > 1:
            # Set the value to 1
            individual.gene[x] = 1
        # Otherwise iof the gene is less than 0
        elif individual.gene[x] < 0:
            # Set the value of the gene to 0
            individual.gene[x] = 0
    # Return the individual
    return individual

# Return an individual according to tournament selection
def tournamentSelection(population, tournament_number):
    # Get a random selection of parents
    parents = []
    # For the number of parents
    for y in range(tournament_number):
        # Add a random parent to the list of parents
        parents.append(population[random.randint(0, len(population)-1)])
    # Choose the first parent as the fittest
    fittest_parent = parents[0]
    # For every parent
    for parent in parents:
        # if the fitness is less than the fittest parents fitness
        if parent.fitness <= fittest_parent.fitness:
            # Make the fittest parent the new parent
            fittest_parent = parent
    # Return the fittest parent from the tournament selection
    return fittest_parent

# Take a population and get a set of offspring
def produceOffspring(population, tournament_number, mutation_chance, mutation_size):
    # Create an offspring array
    offspring = []
    # For every member of the population
    for x in range(len(population)):
        # Get the first parent via tournament selection
        parent1 = tournamentSelection(population, tournament_number)
        # Get the first parent via tournament selection
        parent2 = tournamentSelection(population, tournament_number)
        # Generate a child solution via mutation and crossover
        child = mutate(crossover(parent1, parent2), mutation_chance, mutation_size)
        # Add the new child to the list of offspring
        offspring.append(child)
    # Update the offspring fitness
    updatePopulationFitness(offspring)
    # Pick a random individual to be the worst
    worst = offspring[round(random.random()*len(offspring))-1]
    # For every individual in the population
    for individual in offspring:
        # If its fitness is worst than the current worst
        if individual.fitness > worst.fitness:
            # Set the worst to be the current individual
            worst = individual
    # Replace the worst offspring with the best individual from the previous generation
    offspring[offspring.index(worst)] = findBestIndividual(population)
    # return the offspring array
    return offspring 

# Find the best individual in a population
def findBestIndividual(population):
    # Pick a random individual to be the best
    best = population[round(random.random()*len(population))-1]
    # For every individual in the population
    for individual in population:
        # If its fitness is better than the current best
        if individual.fitness < best.fitness:
            # Set the best to be the current individual
            best = individual
    # Return the best
    return best

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

# Get the average population fitness
def bestPopulationFitness(population):
    # Initialise the total fitness variable
    best_fitness = population[random.randint(0, len(population)-1)].fitness
    # For every individual
    for individual in population:
        # If the individual fitness is better than the best fitness
        if individual.fitness < best_fitness:
            # Set the best fitness to the current individuals fitness
            best_fitness = individual.fitness
    # Return the best of the population's fitness
    return best_fitness

# Evaluate the population fitness
def geneticSolve(population, iterations, tournament_number, mutation_chance, mutation_size):
    # Array to store average population fitness
    average_population_fitness = [averagePopulationFitness(population)]
    # Array to store population fitness
    best_population_fitness = [bestPopulationFitness(population)]
    # For every iteration
    for a in range(iterations):
        # Evaluate the population and produce a new set of offspring
        population = produceOffspring(population, tournament_number, mutation_chance, mutation_size)
        # Add new average population finess
        average_population_fitness.append(averagePopulationFitness(population))
        # Add new best population finess
        best_population_fitness.append(bestPopulationFitness(population))
    # Draw a graph of the results
    plt.plot(average_population_fitness, label = "Average")
    plt.plot(best_population_fitness, label = "Best")
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Minimise Genetic Algorithm with a Population of 50 Individuals')
    plt.legend()
    plt.show()

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
population = initiatePopulation(50, 50)
# Run the genetic algorithm
geneticSolve(population, 50, 2, 0.05, 0.3)
