import math
import random
import time
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################

# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True


######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################

def distance(instance1, instance2, method):
    if method == "euclidean":
        if instance1 is None or instance2 is None:
            return float("inf")
        sumOfSquares = 0
        for i in range(1, len(instance1)):
            sumOfSquares += (instance1[i] - instance2[i])**2
        return math.sqrt(sumOfSquares)
    elif method == "cosine":
        if instance1 is None or instance2 is None:
            return float("inf")
        instance1 = np.array(instance1[1:]).reshape(1, -1)[0]
        instance2 = np.array(instance2[1:]).reshape(1, -1)[0]
        dot_prod = np.dot(instance1, instance2)
        length_prod = np.linalg.norm(instance1) * np.linalg.norm(instance2)
        cosine_dist = (1 - (dot_prod / length_prod))
        return cosine_dist
    else:
        if instance1 is None or instance2 is None:
            return float("inf")
        
        numer = 0
        denom = 0
        for i in range(1, len(instance1)):
            numer += min(instance1[i], instance2[i])
            denom += max(instance1[i], instance2[i])
        return 1 - (numer / denom)

    

def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids, method):
    minDistance = distance(instance, centroids[0], method)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], method)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, labels, centroids, method):
    clusters = createEmptyListOfLists(len(centroids))
    cluster_labels = createEmptyListOfLists(len(centroids))
    
    for instance, label in zip(instances, labels):
        clusterIndex = assign(instance, centroids, method)
        clusters[clusterIndex].append(instance)
        cluster_labels[clusterIndex].append(label)
    return clusters, cluster_labels

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, labels, k, method, animation=False, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    if animation:
        delay = 1.0 # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)

    iteration = 0
    while (centroids != prevCentroids):
        iteration += 1
        clusters, clusters_labels = assignAll(instances, labels, centroids, method)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, method)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)
        
        for i in range(len(clusters_labels)):
                clusters_labels[i] = [int(item[0]) for item in clusters_labels[i]]
        
        true_labels = clusters_labels
        pred_labels = []
        for i, l in enumerate(true_labels):
            majority_dict = {}

            if l == []:
                pred_labels.append([])
                continue

            for entry in l:
                if entry not in majority_dict:
                    majority_dict[entry] = 0
                majority_dict[entry] += 1
            
            majority_label = max(majority_dict.items(), key=lambda x: x[1])[0]
            pred_labels.append(([majority_label] * len(clusters_labels[i])))


    print(iteration)

    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["pred_labels"] = pred_labels
    result["true_labels"] = true_labels

    return result

def computeWithinss(clusters, centroids, method):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, method)**2
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        print("k-means trial %d," % i ,
        trialClustering = kmeans(instances, k))
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering


######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################

def printTable(instances):
    for instance in instances:
        if instance != None:
            line = instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)

def extractAttribute(instances, index):
    result = []
    for instance in instances:
        result.append(instance[index])
    return result

def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc-r, yc-r, xc+r, yc+r, outline=color)

def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc-r, yc-r, xc+r, yc+r, fill=color)

def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / max((maxX - minX), 1)
    scaleY = float(height - 2*margin) / max((maxY - minY), 1)
    for instance in instances:
        x = 5*(random.random()-0.5)+margin+(instance[1]-minX)*scaleX
        y = 5*(random.random()-0.5)+height-margin-(instance[2]-minY)*scaleY
        if (shape == "square"):
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()

def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / max((maxX - minX), 1)
    scaleY = float(height - 2*margin) / max((maxY - minY), 1)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1]-minX)*scaleX
            y1 = height - margin - (p1[2]-minY)*scaleY
            x2 = margin + (p2[1]-minX)*scaleX
            y2 = height - margin - (p2[2]-minY)*scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()

def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result

def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    canvas = Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas

def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)

def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(margin/2, height-margin/2, width-5, height-margin/2,
                       width=2, arrow=LAST)
    canvas.create_text(margin, height-margin/4,
                       text=str(minX), font="Sans 11")
    canvas.create_text(width-margin, height-margin/4,
                       text=str(maxX), font="Sans 11")
    canvas.create_line(margin/2, height-margin/2, margin/2, 5,
                       width=2, arrow=LAST)
    canvas.create_text(margin/4, height-margin,
                       text=str(minY), font="Sans 11", anchor=W)
    canvas.create_text(margin/4, margin,
                       text=str(maxY), font="Sans 11", anchor=W)
    canvas.update()


def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)

def paintDataset2D(canvas, instances):
    canvas.delete(ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()

def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids,
                    "Withinss: %.1f" % withinss)

def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex%len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if (centroid != None):
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width/2, 20, text=title, font="Sans 14")
    canvas.update()


######################################################################
# Test code
######################################################################

#dataset = loadCSV("/Users/yanjiefu/Downloads/tshirts-G.csv")
dataset = loadCSV("C:\\CSE 572\\CSE-572\\HW3\\kmeans_data\\data.csv")
label_data = loadCSV("C:\\CSE 572\\CSE-572\\HW3\\kmeans_data\\label.csv")

#showDataset2D(dataset)
#printTable(clustering["centroids"])


for method in ["euclidean", "cosine", "jaccard"]:
    clustering = kmeans(dataset, label_data, k=10, method=method, animation=False)
    print(f"{method} SSE: ", clustering['withinss'])

    clusters = clustering['clusters']
    clusters_labels = clustering['pred_labels']
    true_labels = clustering['true_labels']

    yhat = []
    y = []
    for pred_sublist, true_sublist in zip(clusters_labels, true_labels):
        yhat.extend(pred_sublist)
        y.extend(true_sublist)

    accuracy = accuracy_score(y, yhat)
    print(f"{method} accuracy: ", accuracy)
    print()


def kmeans_constraint(instances, labels, k, max_iter, method, animation=False, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids 
        random.seed(time.time())
        centroids = random.sample(instances, k)      
    else:
        centroids = initCentroids
    prevCentroids = []
    if animation:
        delay = 1.0 # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    iteration = 0
    priorSSE = 0
    withinss = 0
    while (centroids != prevCentroids and priorSSE >= withinss and iteration < max_iter):
        iteration += 1
        clusters, clusters_labels = assignAll(instances, labels, centroids, method)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids, method)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)
        
        
        for i in range(len(clusters_labels)):
                clusters_labels[i] = [int(item[0]) for item in clusters_labels[i]]

        true_labels = clusters_labels
        pred_labels = []
        for i, l in enumerate(true_labels):
            majority_dict = {}

            if l == []:
                pred_labels.append([])
                continue

            for entry in l:
                if entry not in majority_dict:
                    majority_dict[entry] = 0
                majority_dict[entry] += 1
            
            majority_label = max(majority_dict.items(), key=lambda x: x[1])[0]
            pred_labels.append(([majority_label] * len(clusters_labels[i])))
        
        if priorSSE < withinss and priorSSE != 0:
            break
        priorSSE = withinss
    print(iteration)

    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["pred_labels"] = pred_labels
    result["true_labels"] = true_labels

    return result

for method in ["euclidean", "cosine", "jaccard"]:
    clustering = kmeans_constraint(dataset, label_data, k=10, max_iter=500, method=method, animation=False)
    print(f"{method} SSE (constrained): ", clustering['withinss'])

    clusters = clustering['clusters']
    clusters_labels = clustering['pred_labels']
    true_labels = clustering['true_labels']

    yhat = []
    y = []
    for pred_sublist, true_sublist in zip(clusters_labels, true_labels):
        yhat.extend(pred_sublist)
        y.extend(true_sublist)

    
    accuracy = accuracy_score(y, yhat)
    print(f"{method} accuracy: ", accuracy)
    print()