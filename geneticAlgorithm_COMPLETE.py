# Imports
import random
import math
import copy
import matplotlib.pyplot as plt
import logging
from time import gmtime, strftime

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
    def updateRules(self, testData):
        # Get the length of the rule
        data_length = len(testData[0])
        # Get the length of the rule
        rule_length = (data_length*2)-1
        # Get the total length of the gene
        gene_length = len(self.gene)
        # Empty the list of rules
        self.rules = []
        # create a variable to store the position in the gene
        gene_position = 0
        # Create a variable to hold the current rule
        rule_number = 0
        # While there is enough gene left to make another rule
        while gene_position + rule_length < gene_length:
            # Create an array to store the rule's values
            rule_values = []
            # Create a counter to store the position in the gene
            counter = 0
            # While the counter is less than the length of the rule
            while counter < (data_length*2)-2:
                # If the first value in the pair is bigger than the 2nd value
                if (self.gene[(rule_number*(rule_length))+counter+1] < self.gene[(rule_number*(rule_length))+counter]):
                    # Swap the pair of values in the gene
                    self.gene[(rule_number*(rule_length))+counter], self.gene[(rule_number*(rule_length))+counter+1] = self.gene[(rule_number*(rule_length))+counter+1], self.gene[(rule_number*(rule_length))+counter]
                # Add the 2 values to the rule values
                rule_values.append(self.gene[(rule_number*(rule_length))+counter])
                rule_values.append(self.gene[(rule_number*(rule_length))+counter+1])
                # Increment the counter by 2
                counter += 2
            # Get the rules output from the gene, rounded to the nearest whole number
            rule_output = round(self.gene[(rule_number*data_length)+(data_length*2)-2])
            # Create a rule using the rule values and output from the individual's gene
            new_rule = rule(rule_values, rule_output)
            # Add the newly created rule to the individuals list of rules
            self.rules.append(new_rule)
            # Increment the rule number
            rule_number += 1
            # Increment the position by the rule length
            gene_position += rule_length
    # Test some values against all of the individual's rules
    def test(self, data):
        # Set the result to -1
        result = -1
        # For every rule the individual has
        for rule in self.rules:
            # Test the data against the rule
            result = rule.test(data)
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
        # Set the bounds of the rule
        self.rule_values = rule_values
        # Set the output of the rule
        self.output = output
    # Create a test method
    def test(self, values):
        # Create a boolean to store whether the test has been passed
        passed = True
        # For every value provided (-2 to avoid the output)
        for a in range(len(values)-2):
            # If it doesn't fit within the bound of the test and isn't -1 (missing data)
            if not (values[a] > self.rule_values[a*2] and values[a] < self.rule_values[a*2-1]) and values[a] != -1:
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

# Write data to a file
def writeCSV(CSVdata, fileName):
    # Open a file with the given file name
    f = open("output_data/"+fileName, "a")
    # For every row
    for a in range(len(CSVdata)):
        # Create a string to print
        output_string = ""
        # For every column
        for b in range(len(CSVdata[0])):
            # Add the data to the output string
            output_string += CSVdata[a][b]
        # Write the line of data to the file, removing the last comma and adding a new line
        f.write(output_string+"\n")
    # Close the file
    f.close()

# Write data to a file
def write1DCSV(CSVdata, fileName):
    # Open a file with the given file name
    f = open("output_data/"+fileName, "a")
    # For every row
    for a in range(len(CSVdata)):
        # Write the line of data to the file, removing the last comma and adding a new line
        f.write(str(CSVdata[a])+",")
    # Close the file
    f.close()

# Finess function, tests all the test data against the individual and returns how many tests were correct
def findFitness(individual, testData):
    # Get the test data from the specified data file
    test_list = testData
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
    # Return the fitness of the individual
    return fitness

# Function to import the test data from a text file
def getData(location):
    # Open the given file
    with open(location) as f:
        # Get a list
        test_list = [line.rstrip() for line in f]
    # Create a list to hold parsed values
    parsed_list = []
    # For every item in the list of test data
    for item in test_list:
        # Get the line, parsed by commas - converting each one into a float
        items = [float(x) for x in item.split()]
        # If the data is the 3rd file
        if location == "data/data3.txt":
            # Ignore the first column
            items = items [1:]
        # Add the values to a list of parsed data
        parsed_list.append(items)
    # Return the test data
    return parsed_list

# Update the population's fitness
def updatePopulationFitness(population, testData):
    # For every individual
    for individual in population:
        # Find its fitness
        individual.fitness = findFitness(individual, testData)

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
def initiatePopulation(size, rule_number, min_val, max_val, data, min_rule_number, max_rule_number):
    # Create a population array
    population = []
    # For the given size
    for x in range(size):
        # If the rule number is provided
        if rule_number != None:
            # Calculate the gene size accordingly
            gene_size = (rule_number * (2*len(data[0])))-rule_number
        # Otherwise
        else:
            # Get a random number of rules between the rule limits
            random_rule_number = random.randint(min_rule_number, max_rule_number)
            # Calculate the gene size accordingly
            gene_size = (random_rule_number * (2*len(data[0])))-random_rule_number
        # Create a new individual
        new_indiviual = individual(gene_size, rule_number)
        # Set its genes to random
        new_indiviual.gene = generateRandomGene(gene_size, min_val, max_val)
        # Add the individual to the population
        population.append(new_indiviual)
    # Update the population fitness
    updatePopulationFitness(population, data)
    # Return the population
    return population

# Create a child by crossing over 2 parents
def crossover(parent1, parent2):
    # Get the gene number of the first parent
    size_1 = len(parent1.gene)
    # Get the gene number of the second parent
    size_2 = len(parent2.gene)
    # If the first parent has a bigger gene
    if size_1 > size_2:
        # Get the crossover point from the second parent
        crossover_point = round(random.random()*size_2)
    # otherwise
    else:
        # Get the crossover point from the first parent
        crossover_point = round(random.random()*size_1)
    # Create an offspring array
    offspring = individual(1)
    # Create a potential gene from the first part of the first parent's gene and the second part of the second parent's gene
    potential_gene_1 = parent1.gene[0:crossover_point+1] + parent2.gene[crossover_point+1:len(parent2.gene)+1]
    # Create a potential gene from the first part of the second parent's gene and the second part of the first parent's gene
    potential_gene_2 = parent2.gene[0:crossover_point+1] + parent1.gene[crossover_point+1:len(parent1.gene)+1]
    # Choose randomly between the 2 potential genes for the child
    potential_genes = [potential_gene_1, potential_gene_2]
    # Combine parent1 before the crossover point with parent2 after the crossover point
    offspring.gene = potential_genes[round(random.random())]
    # Return the offspring
    return offspring

# Create a child by crossing over 2 parents
def mutate(individual, mutation_chance, mutation_size, min_val, max_val, rule_mutation_chance, min_rule_number, max_rule_number, rule_length):
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
    # If the rule number should also be mutated
    if rule_mutation_chance != None:
        #If the random value is less than the rule_mutation_chance
        if (random.random() < rule_mutation_chance):
            # With 50% chance
            if round(random.random()) == 1:
                # If adding a new rule wont make the genome longer than the maximum rule number
                if len(individual.gene)+rule_length <= rule_length * max_rule_number:
                    # Add a random value to the gene defined by the mutation size
                    individual.gene += generateRandomGene(rule_length, min_val, max_val)
            # otherwise
            else:
                # If removing a rule from the gene won't make the genome shorter than the minimum size
                if len(individual.gene)-rule_length >= rule_length * min_rule_number:
                    # Subtract a rule from the gene
                    individual.gene = individual.gene[0:len(individual.gene)-rule_length-1]
    # Return the individual
    return individual

# Return an individual according to tournament selection
def tournamentSelection(population, tournament_number, replace_range):
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
        # if the fitness is greater than the fittest parents fitness
        if parent.fitness > fittest_parent.fitness + replace_range:
            # Make the fittest parent the new parent
            fittest_parent = parent
        # Otherwise, if it has the same fitness but a smaller genome
        elif parent.fitness >= fittest_parent.fitness - replace_range and len(parent.gene) < len(fittest_parent.gene):
            # Make the fittest parent the new parent
            fittest_parent = parent
    # Return the fittest parent from the tournament selection
    return fittest_parent

# Take a population and get a set of offspring
def produceOffspring(population, tournament_number, mutation_chance, mutation_size, min_val, max_val, training_data, rule_mutation_chance, min_rule_number, max_rule_number, replace_range):
    # Create an offspring array
    offspring = []
    # Get the rule length (number of genes to make a rule) from the data
    rule_length = (len(training_data[0])*2)-1
    # For every member of the population
    for x in range(len(population)):
        # Get the first parent via tournament selection
        parent1 = tournamentSelection(population, tournament_number, replace_range)
        # Get the first parent via tournament selection
        parent2 = tournamentSelection(population, tournament_number, replace_range)
        # Generate a child solution via mutation and crossover
        child = mutate(crossover(parent1, parent2), mutation_chance, mutation_size, min_val, max_val, rule_mutation_chance, min_rule_number, max_rule_number, rule_length)
        # Update the childs rules
        child.updateRules(training_data)
        # Add the new child to the list of offspring
        offspring.append(child)
    # Update the offspring fitness
    updatePopulationFitness(offspring, training_data)
    # Pick a random individual to be the worst
    worst = offspring[round(random.random()*len(offspring))-1]
    # For every individual in the population
    for individual in offspring:
        # If its fitness is worst than the current worst
        if individual.fitness < worst.fitness:
            # Set the worst to be the current individual
            worst = individual
    # Replace the worst offspring with a deep copy best individual from the previous generation
    offspring[offspring.index(worst)] = copy.deepcopy(findBestIndividual(population))
    #print(f"Top rule no.: {len(findBestIndividual(population).gene)/rule_length}")
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
        # Otherwise, if it has the same fitness but a smaller genome
        elif individual.fitness == best.fitness and len(individual.gene) < len(best.gene):
            # Set the best to be the current individual
            best = individual
    # Return the best
    return best

# Get the average population fitness
def averagePopulationFitness(population):
    # create a blank list to hold the fitnesses
    fitness_list = []
    # Initialise the total fitness variable
    total_fitness = 0
    # For every individual
    for individual in population:
        # Add the fitness to the total fitness
        total_fitness += individual.fitness
        # Append the fitness value to the fitness list
        fitness_list.append(individual.fitness)
    # Return a mean average of the population's fitness
    return total_fitness/len(population)

# Get the average number of rules per individual
def averageRuleNumber(population, rule_length):
    # Initialise the total fitness variable
    total_rules = 0
    # For every individual
    for individual in population:
        # Add the floored rule number to the total number of rules
        total_rules += math.floor(len(individual.gene)/rule_length)
    # Return a mean average of the population's fitness
    return total_rules/len(population)

# Train the genetic algorithm with the provided data and return the best solution
def trainAlgorithm(population, iterations, tournament_number, mutation_chance, mutation_size, min_val, max_val, training_data, rule_mutation_chance, min_rule_number, max_rule_number, replace_range):
    # Get the rule length (number of genes to make a rule) from the data
    rule_length = (len(training_data[0])*2)-1
    # Array to store average population fitness
    average_population_fitness = [averagePopulationFitness(population)]
    # Array to store population fitness
    best_population_fitness = [findBestIndividual(population).fitness]
    # Calculate the average number of rules in the population
    average_rule_numbers = [averageRuleNumber(population, rule_length)]
    # For every iteration
    for a in range(iterations):
        # Evaluate the population and produce a new set of offspring
        population = produceOffspring(population, tournament_number, mutation_chance, mutation_size, min_val, max_val, training_data, rule_mutation_chance, min_rule_number, max_rule_number, replace_range)
        # Add new average population finess
        average_population_fitness.append(averagePopulationFitness(population))
        # Add new best population finess
        best_population_fitness.append(findBestIndividual(population).fitness)
        # Add new average rule number
        average_rule_numbers.append(averageRuleNumber(population, rule_length))
    # Return a tuple with the best performing individual from training, an array of the average population fitness and an array of the best population fitness
    return (findBestIndividual(population), average_population_fitness, best_population_fitness, average_rule_numbers)

# Split a list of data into k parts
def kFoldSplit(data, k):
    # Create a list to hold the split data
    split_data = []
    # Calculate the interval size
    interval_size = int(len(data)/k)
    # Iterate through the list with calculated interval
    for x in range (0, len(data), interval_size):
        # For each set, add the values between x and x+interval to the output
        split_data.append(data[x:x + interval_size])
    # Return the split data
    return split_data

# Evaluate the population fitness
def geneticSolve(population_size, rule_number, min_val, max_val, iterations, tournament_number, mutation_chance, mutation_size, data_location, fold_number, rule_mutation_chance, min_rule_number, max_rule_number, repeat_times, file_name, replace_range):
    # Get the data from the specified file
    data = getData(data_location)
    # Print out information
    print(f"\nBreaking data into {fold_number} segments\nPopulation size: {population_size}\nRule number: {rule_number}\nIterations: {iterations}")
    # Create an array to output to the file
    output_array = []
    # If the data should be segmented and tested
    if fold_number != 1 :
        # Add title to the output array
        output_array.append((" ,","Training,","Testing,"))
        # For every time the algorithm should repeat
        for round in range(repeat_times):
            # Create an array to store every segment's test fitness
            test_fitnesses = []
            # Create an array to store every segment's training fitness
            training_fitnesses = []
            # Split the data into the required number of folds
            folded_data = kFoldSplit(data, fold_number)
            # For every section of the data
            for segment_number, segment in enumerate(folded_data):
                # Copy the fold and use it as the test data
                test_data = copy.deepcopy(segment)
                # Copy the data and remove the test data to use as training data
                training_data = []
                # For every section of the data
                for training_segment_number, training_segment in enumerate(folded_data):
                    # If the current section isnt the test data
                    if segment_number != training_segment_number:
                        # Add it to the training data
                        training_data += training_segment
                # Print out information
                print("\nTraining poulation")
                # Initiate the population
                population = initiatePopulation(population_size, rule_number, min_val, max_val, training_data, min_rule_number, max_rule_number)
                # Get the best individual, average fitness and best fitness from training using the training data
                result = trainAlgorithm(population, iterations, tournament_number, mutation_chance, mutation_size, min_val, max_val, training_data, rule_mutation_chance, min_rule_number, max_rule_number, replace_range)
                # Test the best solution with the test data
                test_result = findFitness(result[0], test_data)
                # Add the fitness to the list of fitnesses
                test_fitnesses.append(test_result)
                training_fitnesses.append(result[0].fitness)
                # Print out information
                print(f"Training fitness: {result[0].fitness}/{len(training_data)}\nTest fitness: {test_result}/{len(test_data)}")
            # Print out information
            print(f"\nTotal fitness: {(sum(test_fitnesses))}/{len(data)}")
            # Calculate the average training fitness for the round
            average_training_fitness = sum(training_fitnesses)/(len(training_data)*fold_number)
            # Calculate the average testing fitness for the round
            average_test_fitness = sum(test_fitnesses)/len(data)
            # Add the data from training/testing round to the output array
            output_array.append((f"{round},", f"{average_training_fitness},", f"{average_test_fitness}"))
        # Write the results to a CSV file
        writeCSV(output_array, file_name)
    # Otherwise, if there is only one segment (no k fold testing)
    else:
        # Add title to the output array
        output_array.append((" ,","Training,","Rule Number,"))
        # Create an array to hold the results
        results = []
        # For every time the algorithm should repeat
        for round in range(repeat_times):
            # Initiate the population
            population = initiatePopulation(population_size, rule_number, min_val, max_val, data, min_rule_number, max_rule_number)
            # Print out information
            print("\nTraining poulation")
            # Get the best individual, average fitness and best fitness from training
            result = trainAlgorithm(population, iterations, tournament_number, mutation_chance, mutation_size, min_val, max_val, data, rule_mutation_chance, min_rule_number, max_rule_number, replace_range)
            # Print out information
            print(f"Training fitness: {result[0].fitness}/{len(data)}")
            # Get the rule length (number of genes to make a rule) from the data
            rule_length = (len(data[0])*2)-1
            # Add the data from training round to the output array
            output_array.append((f"{round},", f"{result[0].fitness/len(data)},", f"{len(result[0].gene)/rule_length},"))
            # Add the results to the list of resutls
            results.append(result)
        # Get an array to hold the averages of all the rounds
        average = []
        # Get an array to hold the bests of all the rounds
        best = []
        # Get an array to hold the rule numbers of all the rounds
        rules = []
        # For each epoch
        for a in range(len(results[0][1])):
            # Create a variable to store a sum for the average
            average_sum = 0
            # Create a variable to store a sum for the best
            best_sum = 0
            # Create a variable to store a sum for the best
            rule_sum = 0
            # For every round
            for b in range(len(results)):
                # Add the average from the round to the average sum
                average_sum += results[b][1][a]
                # Add the best from the round to the best sum
                best_sum += results[b][2][a]
                # Add the best from the round to the best sum
                rule_sum += results[b][3][a]
            # Divide the sum by the number of rounds and append it to the list of average values
            average.append(average_sum/len(results))
            # Divide the sum by the number of rounds and append it to the list of best values
            best.append(best_sum/len(results))
            # Divide the sum by the number of rounds and append it to the list of best values
            rules.append(rule_sum/len(results))
        # Write the results to a CSV file
        #writeCSV(output_array, file_name)
        #
        #write1DCSV(best, "1d_GA.csv")
        # Draw a graph of the results
        plt.plot(average, label = "Average")
        plt.plot(best, label = "Best")
        #plt.plot(rules, label = "Rule Number")
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.title(f'Rule Learning with a Population of {population_size}, mutation chance {mutation_chance} and a mutation amount of {mutation_amount}')
        plt.legend()
        plt.show()

# Print a new line
print()

# Define the variables for the algorithm
max_val = 1
min_val = 0
population_size = 100
generations = 200
rule_number = 2
tournament_number = 2
mutation_chance = 0.04
mutation_amount = 0.9
rule_mutation_chance = 0
min_rule_number = 1
max_rule_number = 40
fold_number = 1
repeat_times = 10
rule_size_factor = 0.07
data_location = "data/data1.txt"



# Get today's date and time
today = strftime("%Y-%m-%d %H:%M:%S", gmtime())
# get the name of the data file
input_data_file = data_location[data_location.index("/")+1:data_location.index(".")]
# Create a file name using the variables from the setup
file_name = f"{input_data_file}/{population_size} I, {generations} E, {rule_number} RN, {mutation_chance} MC, {mutation_amount} MS, {fold_number} FN ({today}).csv"

# Run the genetic algorithm
geneticSolve(population_size, rule_number, min_val, max_val, generations, tournament_number, mutation_chance, mutation_amount, data_location, fold_number, rule_mutation_chance, min_rule_number, max_rule_number, repeat_times, file_name, population_size*rule_size_factor)