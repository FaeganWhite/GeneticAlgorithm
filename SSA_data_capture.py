from __future__ import division
from __future__ import print_function
import sys

from numpy.ma import average
if sys.version.startswith('3'):
    xrange = range
import numpy as np
rand = np.random.rand
from scipy.spatial.distance import pdist, squareform
import time
import matplotlib.pyplot as plt
import copy

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

data = getData("data/data1.txt")
graph = []
graph_max_fitness = []
solutions = []

def problem(x):
    #print()
    #
    spider_fitness = []

    spider_rule_count = []
    #
    if graph == [] :
        #
        graph_max_fitness.append(0)
        #
        graph.append(0)
    #
    for spider_no, spider in enumerate(x):
        #
        bounded_x = []
        #
        #print(spider)
        #
        for value in spider:
            #
            if value <= 0:
                #
                bounded_x.append(0)
            #
            elif value >= 1:
                #
                bounded_x.append(1)
            #
            else:
                #
                bounded_x.append(value)
        #
        rules_passed = 0
        #
        for piece in data:
            #
            pos = 0
            #
            result = -1
            #
            while pos < len(bounded_x):
                # Create a boolean to store whether the test has been passed
                passed = True
                # For every value provided (-2 to avoid the output)
                for a in range(len(piece)-1):
                    #print(f"data at {a}, rule at {pos+((a*2)+1)}")
                    # If it doesn't fit within the bound of the test and isn't -1 (missing data)
                    if not (bounded_x[pos+(a*2)] <= bounded_x[pos+((a*2)+1)] and piece[a] >= bounded_x[pos+(a*2)] and piece[a] <= bounded_x[pos+((a*2)+1)]) and not (bounded_x[pos+(a*2)] > bounded_x[pos+((a*2)+1)] and piece[a] <= bounded_x[pos+(a*2)] and piece[a] >= bounded_x[pos+((a*2)+1)]):
                        #print("break")
                        # Mark the test as failed
                        passed = False
                        # Break the test loop
                        break
                # If the test was passed
                if passed is True:
                    #
                    result = round(bounded_x[pos+(len(piece)*2)-2])
                    #print(f"result from {pos+(len(piece)*2)-2}")
                    #print("passed")
                    #
                    break
                #
                pos += (len(piece)*2)-1
            #
            #print(f"comparing {result} with {piece[-1]} at position {len(piece)-1}")
            if result == piece[-1]:
                #
                #print("winner winner")
                # Return the output of the rule
                rules_passed += 1
                #if pos == 17:
                #    time.sleep(1000)
        #
        spider_fitness.append(len(data)-rules_passed)

        spider_rule_count.append(rules_passed)
        #print(spider_fitness)
        
        #print(bounded_x)
        #time.sleep(1)
    graph.append(average(spider_rule_count))
    #
    if max(spider_rule_count) > graph_max_fitness[-1]:
        #
        graph_max_fitness.append(max(spider_rule_count))
        #
    else:
        #
        graph_max_fitness.append(graph_max_fitness[-1])
    #
    print(graph_max_fitness[-1])
    #print(spider_fitness)
    np_array = np.asarray(spider_fitness, dtype=np.float64)
    #
    #print(np_array)
    #time.sleep(1)
    #print(np.sum(x**2, 1))
    #print(type(np_array))
    return np_array

class SSA(object):
    def __init__(self, func, 
                 dim = 65,
                 bound = 10,
                 max_iteration = 1000,
                 pop_size = 100,
                 r_a = 1,
                 p_c = 0.7,
                 p_m = 0.5):
        self.func = func
        self.dim = dim
        self.bound = bound
        self.max_iteration = max_iteration
        self.pop_size = pop_size
        self.r_a = r_a
        self.p_c = p_c
        self.p_m = p_m
            
    def run(self, show_info = False):
        self.g_best = np.Inf
        self.g_best_hist = []
        self.g_best_pos = np.zeros(self.dim)
        self.position = rand(self.pop_size, self.dim) * 2 * self.bound - self.bound
        target_position = self.position.copy()
        target_intensity = np.zeros(self.pop_size)
        mask = np.zeros((self.pop_size, self.dim))
        movement = np.zeros((self.pop_size, self.dim))
        inactive = np.zeros(self.pop_size)
        
        if show_info:
            import datetime, time
            print(" " * 15 + "SSA starts at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("=" * 62)
            print(" iter    optimum    pop_min  base_dist  mean_dist time_elapsed")
            print("=" * 62)
            self.start_time = time.process_time()
        
        iteration = 0
        while (iteration < self.max_iteration):
            iteration += 1
            spider_fitness = self.func(self.position)
            base_distance = np.mean(np.std(self.position, 0))
            distance = squareform(pdist(self.position, 'cityblock'))
            
            if np.min(spider_fitness) < self.g_best:
                self.g_best = np.min(spider_fitness)
                self.g_best_pos = self.position[np.argmin(spider_fitness)].copy()
            self.g_best_hist.append(self.g_best)
            if show_info and (iteration == 1 or iteration == 10
                    or (iteration < 1001 and iteration % 100 == 0) 
                    or (iteration < 10001 and iteration % 1000 == 0)
                    or (iteration < 100000 and iteration % 10000 == 0)):
                elapsed_time = time.process_time() - self.start_time
                print(repr(iteration).rjust(5), self.g_best, "%.4e" % np.min(spider_fitness),
                      "%.4e" % base_distance, "%.4e" % np.mean(distance), 
                      "%02d:%02d:%02d.%03d" % (elapsed_time // 3600, elapsed_time // 60 % 60, 
                                               elapsed_time % 60, (elapsed_time % 1) * 1000))
            
            intensity_source = np.log(1. / (spider_fitness + 1E-100) + 1)
            intensity_attenuation = np.exp(-distance / (base_distance * self.r_a))
            intensity_receive = np.tile(intensity_source, self.pop_size).reshape(self.pop_size, self.pop_size) * intensity_attenuation
            
            max_index = np.argmax(intensity_receive, axis = 1)
            keep_target = intensity_receive[np.arange(self.pop_size),max_index] <= target_intensity
            keep_target_matrix = np.repeat(keep_target, self.dim).reshape(self.pop_size, self.dim)
            inactive = inactive * keep_target + keep_target
            target_intensity = target_intensity * keep_target + intensity_receive[np.arange(self.pop_size),max_index] * (1 - keep_target)
            target_position = target_position * keep_target_matrix + self.position[max_index] * (1 - keep_target_matrix)
            
            rand_position = self.position[np.floor(rand(self.pop_size * self.dim) * self.pop_size).astype(int), \
                np.tile(np.arange(self.dim), self.pop_size)].reshape(self.pop_size, self.dim)
            new_mask = np.ceil(rand(self.pop_size, self.dim) + rand() * self.p_m - 1)
            keep_mask = rand(self.pop_size) < self.p_c**inactive
            inactive = inactive * keep_mask
            keep_mask_matrix = np.repeat(keep_mask, self.dim).reshape(self.pop_size, self.dim)
            mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask
                            
            follow_position = mask * rand_position + (1 - mask) * target_position
            movement = np.repeat(rand(self.pop_size), self.dim).reshape(self.pop_size, self.dim) * movement + \
                (follow_position - self.position) * rand(self.pop_size, self.dim)
            self.position = self.position + movement
            
        if show_info:
            elapsed_time = time.process_time() - self.start_time
            print("=" * 62)
            print(repr(iteration).rjust(5), "%.4e" % self.g_best, "%.4e" % np.min(spider_fitness),
                  "%.4e" % base_distance, "%.4e" % np.mean(distance), 
                  "%02d:%02d:%02d.%03d" % (elapsed_time // 3600, elapsed_time // 60 % 60, 
                                           elapsed_time % 60, (elapsed_time % 1) * 1000))
            print("=" * 62)
        return {'global_best_fitness': self.g_best,
                'global_best_solution': self.g_best_pos,
                'iterations': iteration + 1}

epochs = 10

# Create a file name using the variables from the setup
file_name = "spiders/spider_data.csv"


if __name__ == '__main__':
    
    # Create an array to output to the file
    output_array = []
    # Add title to the output array
    output_array.append((" ,","Training,"))
    #
    average_fitness = []
    #
    average_max_fitness = []
    #
    for e in range(epochs):
        #
        graph = []
        #
        graph_max_fitness = []
        #
        SSA(problem).run(True)
        #
        average_fitness.append(copy.deepcopy(graph))
        #
        average_max_fitness.append(copy.deepcopy(graph_max_fitness))
        #
        output_array.append((f"{e},", f"{graph_max_fitness[-1]},"))
    #
    best = []
    #
    average_baby = []
    #
    for val in range(len(average_max_fitness[0])):
        #
        sum = 0
        #
        average_sume = 0
        #
        for num in range(len(average_max_fitness)):
            #
            sum += average_max_fitness[num][val]
            #
            average_sume += average_fitness[num][val]
        #
        best.append(sum/len(average_max_fitness))
        #
        average_baby.append(average_sume/len(average_max_fitness))
    #
    write1DCSV(best, "spiders/best_average.csv")
        
    # Write the results to a CSV file
    writeCSV(output_array, file_name)
    #
    plt.plot(average_baby, label = f"Average")
    plt.plot(best, label = f"Best")
    #
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title(f'Spiders baby!')
    plt.legend()
    plt.show()

