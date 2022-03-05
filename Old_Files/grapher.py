# Import required files
import os

# Print out a graph grid
def printGraph(grid):
    # Print a newline
    print()
    # For every row in the grid
    for row in grid:
        # For every character in the row
        for char in row:
            # Print the character without creating a new line
            print(char, end = '')
        # Print a newline
        print()

# Create a grid filled with whitespace
def createBlankGrid(width, height):
    # Create an empty list to store the output grid
    output_grid = []
    # For the height
    for a in range(height):
        # Create a new list to hold the row information
        new_row = []
        # For every column
        for b in range(width):
            # Add some whitespace to the row
            new_row.append(" ")
        # Add the row to the grid
        output_grid.append(new_row)
    # Return the output grid
    return output_grid

# Add the axis titles to the grid
def addAxisTitles(grid, xLabel, yLabel):
    # Copy the input grid to the output grid
    output_grid = grid
    # Calculate the first x position of the x label
    xStart = round((len(grid[0])/2)-len(xLabel)/2)
    # Calculate the first y position of the y label
    yStart = round((len(grid)/2)-len(yLabel)/2)
    # For every character in the x label
    for x in range(len(xLabel)):
        # Add the label character to the output grid
        output_grid[len(grid)-1][xStart+x] = xLabel[x]
    # For every character in the y label
    for y in range(len(yLabel)):
        # Add the label character to the output grid
        output_grid[yStart+y][0] = yLabel[y]
    # Return the output grid
    return output_grid

# Add the axis to the graph
def addAxis(grid):
    # Copy the input grid to the output grid
    output_grid = grid
    # For every character in the x label
    for x in range(len(output_grid[0])):
        # If the y axis hasn't been reached
        if x > 2:
            # Add the label character to the output grid
            output_grid[len(grid)-3][x] = "_"
    # For every character in the y label
    for y in range(len(output_grid)):
        # If the x axis hasn't been reached
        if y < len(grid)-2:
            # Add the label character to the output grid
            output_grid[y][3] = "|"
    # Return the output grid
    return output_grid

# Add data points to the graph
def addData(data, graph, xRange, yRange, point="*"):
    # Duplicate the input graph to an output graph
    output_graph = graph
    # For every data point
    for a in range(len(data)):
        # Calculate the x position
        x = 3+round(((a+1)/xRange)*(len(graph[0])-4))
        # Calculate the y position
        y = (len(output_graph)-3)-round((data[a]/yRange)*(len(graph)-3))
        # Add the point ton the graph
        output_graph[y][x] = point
    # Return the graph with points
    return output_graph

# Add the scales to the graph
def addScale(graph, xRange, yRange, xGap, yGap):
    # Duplicate the input graph to an output graph
    output_graph = graph
    # Create an array to hold the x scale
    xScale = [0]
    # Create an array to hold the y scale
    yScale = [0]
    # Set the first value on the x axis
    current_value = 0
    # While the xScale is smaller than the xRange
    while xScale[len(xScale)-1] < xRange:
        # Add the gap
        current_value += xGap
        # Add the value to the x scale array
        xScale.append(current_value)
    # Reset the current value to 0
    current_value = 0
    # While the yScale is smaller than the yRange
    while yScale[len(yScale)-1] < yRange:
        # Increment the current value by the y gap
        current_value += yGap
        # Add the current value to the y scale array
        yScale.append(current_value)
    # For every point on the x scale
    for point in xScale:
        # For every character in the scale value
        for a in range(len(str(point))):
            # Add it to the graph
            output_graph[len(graph)-2][3+round((point/xRange)*(len(graph[0])-5))+a] = str(point)[a]
    # For every point on the y scale
    for point in yScale:
        # For every character in the scale value
        for a in range(len(str(point))):
            # Add it to the graph
            output_graph[(len(output_graph)-3)-round((point/yRange)*(len(graph)-3))][1+a] = str(point)[a]
    # Return the output graph
    return output_graph

# Function to graph an array
def drawGraph(data, xLabel="", yLabel="", xGap=1, yGap=2, xRange=0, yRange=0):
    # If the xRange is unspecified
    if xRange == 0:
        # Use the number of data points
        xRange = len(data)
    # If the yRange is unspecified
    if yRange == 0:
        # Use the maximum value
        yRange = max(data)
    # Get the columns and rows of the console
    columns, rows = os.get_terminal_size(0)
    # Calculate the height of the graph
    height = rows-1
    # Calculate the width of the graph
    width = columns-1
    # Create a variable to store the final output
    output_grid = createBlankGrid(width, height)
    # Add the axis to the graph
    output_grid = addAxis(output_grid)
    # Add the titles to the axis
    output_grid = addAxisTitles(output_grid, xLabel, yLabel)
    # Add the scale to the axis
    output_grid = addScale(output_grid, xRange, yRange, xGap, yGap)
    # Add the data to the graph
    output_grid = addData(data, output_grid, xRange, yRange)
    # Print the graph
    printGraph(output_grid)

