# Imports
import random
import math
import copy
import matplotlib.pyplot as plt

# A class for an individual solution
class individual:
    # Initiate the self with v values per rule and r rules (automatically 0)
    def __init__(self, v, r = 0):
        # Create some empty genes (2 for each value plus 1 for the output for every rule)
        self.gene = [0]*v
        # Create a variable to store the fitness of te individual solution
        self.fitness = 0
        # Create a list to hold all of the individuals rules
        self.rules = []
    # Update all of the rule objects owned by the individual from its genes
    def updateRules(self):
        # Get the length of the rule
        rule_length = len(getTestData(data_location)[0])
        # Get the total length of the gene
        gene_length = len(self.gene)
        # Get the number of rules from the gene
        rule_number = round(gene_length/((rule_length*2)-1))
        # Empty the list of rules
        self.rules = []
        # For every rule to be created
        for x in range(rule_number):
            # Create an array to store the rule's values
            rule_values = []
            # Create a counter to store the position in the gene
            counter = 0
            # While the counter is less than the length of the rule
            while counter < (rule_length*2)-2:
                # Get the rules value from the gene
                rule_values.append(self.gene[(x*rule_number)+counter])
                # Increment the counter
                counter += 1
            # Get the rules output from the gene, rounded to the nearest whole number
            rule_output = round(self.gene[(x*rule_length)+(rule_length*2)-2])
            #############
            #print("Gene:", self.gene)
            #print("Gene length:", len(self.gene))
            ##############
            #print("Creating rule:", rule_values, rule_output)
            #print("Rule length:", len(rule_values)+1)
            # Create a rule using the rule values and output from the individual's gene
            new_rule = rule(rule_values, rule_output)
            # Add the newly created rule to the individuals list of rules
            self.rules.append(new_rule)
    # Test some values against all of the individual's rules
    def test(self, data):
        # Set the result to -1
        result = -1
        # For every rule the individual has
        for rule in self.rules:
            # Test the data against the rule
            result = rule.test(data)
            ################
            #print(result)
            # If the rule passed
            if result != -1:
                # Return the result
                return result
        # Otherwise return -1
        return result
    
# A class for an individual solution
class rule:
    # Initiate the self
    def __init__(self, rule_values, output):
        # For every value in the value list
        #for count, value in enumerate(rule_values):
            # If the value is the first in the pair
        #    if count % 2 == 0:
                # If the value is larger than the 2nd value (making the rule invalid)
        #        if value > rule_values[count+1]:
                    # Return false (invalid rule)
        #            return False
        # Set the bounds of the rule
        self.rule_values = rule_values
        # Set the output of the rule
        self.output = output
        # Return true (valid rule)
        return True
    # Create a test method
    def test(self, values):
        # Create a boolean to store whether the test has been passed
        passed = True
        # For every value provided (-2 to avoid the output)
        for a in range(len(values)-2):
            ########
            #print(type(values[a]))
            #print(type(self.rule_values[a*2]))
            #print(self.rule_values[a*2-1])
            # If it doesn't fit within the bound of the test
            if not (values[a] > self.rule_values[a*2] and values[a] < self.rule_values[a*2-1]):
                # Mark the test as failed
                passed = False
                # Break the test loop
                break
        # If the test was passed
        if passed is True:
            # Return the output of the rule
            return self.output
        # Otherwise
        else:
            # Return -1
            return -1

# Finess function, tests all the test data against the individual and returns how many tests were correct
def findFitness(individual):
    # Get the test data from the specified data file
    test_list = getTestData(data_location)
    # Create a fitness variable, initilaising it as 0
    fitness = 0
    # For every value in the test list
    for test_data in test_list:
        # Get the result from testing it against the individuals rules
        result = individual.test(test_data)
        # If the result matches the result in the test data
        if result == test_data[len(test_data)-1]:
            # Increment the fitness
            fitness +=1
    ######
    #print(fitness)
    # Return the fitness of the individual
    return fitness

# Function to import the test data from a text file
def getTestData(location):
    # Open the given file
    with open(location) as f:
        # Get a list
        test_list = [line.rstrip() for line in f]
    # Create a list to hold parsed values
    parsed_list = []
    # For every item in the list of test data
    for item in test_list:
        # Parse the item by comma and then add the values to a list of parsed data, converting each one to a float
        parsed_list.append([float(x) for x in item.split()])
    # Return the test data
    return parsed_list

# Update the population's fitness
def updatePopulationFitness(population):
    # For every individual
    for individual in population:
        # Find its fitness
        individual.fitness = findFitness(individual)

# Function to generate a random gene
def generateRandomGene(n, min_val, max_val):
    # Create an empty array
    gene = []
    # For n times
    for x in range(n):
        # Generate a random binary value
        random_value = random.uniform(min_val, max_val)
        # Add a value to the gene
        gene.append(random_value)
    # Return the gene
    return gene

# Initiate and return a population of random individuals
def initiatePopulation(size, rule_number, min_val, max_val):
    #
    gene_number = (rule_number * (2*len(getTestData(data_location)[0])))-rule_number
    print("Number of genes:", gene_number)
    print("Size of test data:", len(getTestData(data_location)[0]))
    # Get the number of rules using the rule size and
    rule_number = rule_number
    print("Number of rules", rule_number)
    # Create a population array
    population = []
    # For the given size
    for x in range(size):
        # Create a new individual
        new_indiviual = individual(gene_number, rule_number)
        # Set its genes to random
        new_indiviual.gene = generateRandomGene(gene_number, min_val, max_val)
        # Add the individual to the population
        population.append(new_indiviual)
    # Update the population fitness
    updatePopulationFitness(population)
    # Return the population
    return population

# Create a child by crossing over 2 parents
def crossover(parent1, parent2):
    ########
    #print("Crossingover",parent1.fitness,parent2.fitness)
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
def mutate(individual, mutation_chance, mutation_size, min_val, max_val):
    # For every value in the gene
    for x in range(len(individual.gene)):
        # If the random value is less than the mutation_chance
        if (random.random() < mutation_chance):
            # With 50% chance
            if round(random.random()) == 1:
                # Add a random value to the gene defined by the mutation size
                individual.gene[x] += random.uniform(0, mutation_size)
            # otherwise
            else:
                # Subtract a random value from the gene defined by the mutation size
                individual.gene[x] -= random.uniform(0, mutation_size)
        # If the individual gene is greater than 1
        if individual.gene[x] > max_val:
            # Set the value to 1
            individual.gene[x] = max_val
        # Otherwise iof the gene is less than 0
        elif individual.gene[x] < min_val:
            # Set the value of the gene to 0
            individual.gene[x] = min_val
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
def produceOffspring(population, tournament_number, mutation_chance, mutation_size, min_val, max_val):
    # Create an offspring array
    offspring = []
    # For every member of the population
    for x in range(len(population)):
        # Get the first parent via tournament selection
        parent1 = tournamentSelection(population, tournament_number)
        # Get the first parent via tournament selection
        parent2 = tournamentSelection(population, tournament_number)
        # Generate a child solution via mutation and crossover
        child = mutate(crossover(parent1, parent2), mutation_chance, mutation_size, min_val, max_val)
        # Update the childs rules
        child.updateRules()
        ##############
        ##print(child.gene)
        # Add the new child to the list of offspring
        offspring.append(child)
    ############
    print()
    # Update the offspring fitness
    updatePopulationFitness(offspring)
    # Pick a random individual to be the worst
    worst = offspring[round(random.random()*len(offspring))-1]
    # For every individual in the population
    for individual in offspring:
        # If its fitness is worst than the current worst
        if individual.fitness < worst.fitness:
            # Set the worst to be the current individual
            worst = individual
    print("Replacing individual with fitness",worst.fitness)
    # Replace the worst offspring with a deep copy best individual from the previous generation
    offspring[offspring.index(worst)] = copy.deepcopy(findBestIndividual(population))
    # return the offspring array
    return offspring 

# Find the best individual in a population
def findBestIndividual(population):
    # Pick a random individual to be the best
    best = population[round(random.random()*len(population))-1]
    # For every individual in the population
    for individual in population:
        # If its fitness is better than the current best
        if individual.fitness > best.fitness:
            # Set the best to be the current individual
            best = individual
    print(best.fitness)
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

# Evaluate the population fitness
def geneticSolve(population, iterations, tournament_number, mutation_chance, mutation_size, min_val, max_val):
    # Array to store average population fitness
    average_population_fitness = [averagePopulationFitness(population)]
    # Array to store population fitness
    best_population_fitness = [findBestIndividual(population).fitness]
    # For every iteration
    for a in range(iterations):
        # Evaluate the population and produce a new set of offspring
        population = produceOffspring(population, tournament_number, mutation_chance, mutation_size, min_val, max_val)
        # Add new average population finess
        average_population_fitness.append(averagePopulationFitness(population))
        # Add new best population finess
        best_population_fitness.append(findBestIndividual(population).fitness)
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

print()
# Define the variables for the algorithm
max_val = 1
min_val = 0
population_size = 50
generations = 2000
rule_number = 10
parent_number = 2
mutation_chance = 0.05
mutation_amount = 2
data_location = "data/data1.txt"

# Initiate the population
population = initiatePopulation(population_size, rule_number, min_val, max_val)
# Run the genetic algorithm
geneticSolve(population, generations, parent_number, mutation_chance, mutation_amount, min_val, max_val)
