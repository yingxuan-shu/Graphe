# --coding:utf8--
import cv2
import numpy as np
import operator
from numpy import *


# 1. Read images
def readImage(fileName):
    # Read color image
    img = cv2.imread(fileName, cv2.IMREAD_COLOR)
    return img


# 2. Filters and threshold
def hand_threshold(frame):
    # Blur the image
    blur = cv2.blur(frame, (3, 3))
    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))
    cv2.imwrite('./result/binaryImage.png', mask2)
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    erosion = cv2.erode(dilation, kernel_square, iterations=1)
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)
    return thresh


# 3. Find hand contour
def hand_contour_find(contours, frame):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        # Largest area contour
        h_contour=contours[largest_contour]
        # Find convex hull
        hull = cv2.convexHull(h_contour)
        # Find convex defects
        hull2 = cv2.convexHull(h_contour, returnPoints=False)
        defects = cv2.convexityDefects(h_contour, hull2)

        # Get defect points and draw them in the original image
        # A list of start defect point, end defect point and far defect point
        StartDefect = []
        EndDefect = []
        FarDefect = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(h_contour[s][0])
            end = tuple(h_contour[e][0])
            far = tuple(h_contour[f][0])
            StartDefect.append(start)
            EndDefect.append(end)
            FarDefect.append(far)
            cv2.line(frame, start, far, [255, 0, 100], 2)
            cv2.line(frame, far, end, [255, 0, 100], 2)
        return True, h_contour, StartDefect, EndDefect, FarDefect

# 4. Find Central Mass
def calculeCentralMass(h_contour, FarDefect, frame):
    # Find moments of the largest contour
    moments = cv2.moments(h_contour)

    # Central mass of first order moments
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)
    FarDefect.append(centerMass)
    # Add Central Mass in the end of FarDefect

    # Draw center mass
    cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # add text 'CentralMass' in image
    # cv2.putText(frame, 'CentralMass', tuple(centerMass), font, 1, (255, 255, 255), 2)

    return FarDefect, centerMass, moments


# 5. Calcule distance between far defects to start nodes and end nodes
def calculeDistance(StartDefect, EndDefect, FarDefect, centerMass):
    # Distance from start defect to far node
    distanceBetweenDefectsToStart = []
    for i in range(0, len(StartDefect)):
        startNode = np.array(StartDefect[i])
        farNode = np.array(FarDefect[i])
        distance = np.sqrt(np.power(startNode[0]-farNode[0], 2) + np.power(startNode[1]-farNode[1], 2))
        distanceBetweenDefectsToStart.append(distance)

    # Distance from far defect to end node
    distanceBetweenDefectsToEnd = []
    for i in range(0, len(StartDefect)):
        endNode = np.array(EndDefect[i])
        farNode = np.array(FarDefect[i])
        distance = np.sqrt(np.power(endNode[0] - farNode[0], 2) + np.power(endNode[1] - farNode[1], 2))
        distanceBetweenDefectsToEnd.append(distance)

    # Distance from end node to next start node
    distanceBetweenEndNodeToNextStartNode = []
    for i in range(0, len(StartDefect)):
        endNode = np.array(EndDefect[i])
        nestStartNode = np.array(FarDefect[i+1])
        distance = np.sqrt(np.power(endNode[0] - nestStartNode[0], 2) + np.power(endNode[1] - nestStartNode[1], 2))
        distanceBetweenEndNodeToNextStartNode.append(distance)

    return distanceBetweenDefectsToStart, distanceBetweenDefectsToEnd, distanceBetweenEndNodeToNextStartNode


# 6. Create a graph
def creatGraph(StartDefect, EndDefect, FarDefect, frame, distanceBetweenDefectsToStart, distanceBetweenDefectsToEnd, distanceBetweenEndNodeToNextStartNode):
    # write nodes in a list
    node_list = []
    # write the edges in a list
    edge_list = []

    for i in range(0, len(StartDefect)):
        if distanceBetweenDefectsToStart[i] > 20 and distanceBetweenDefectsToEnd > 20 and distanceBetweenEndNodeToNextStartNode > 30 :
            # write start nodes in startNodeList
            startNodeList = []
            startNode = np.array(StartDefect[i])
            # start node Id
            startNodeList.append(len(node_list))
            # start node l'axe x
            startNodeList.append(startNode[0])
            # start node l'axe y
            startNodeList.append(startNode[1])
            # add start node in the node list
            node_list.append(startNodeList)
            # draw start node in frame
            cv2.circle(frame, StartDefect[i], 10, [255, 0, 100], 3)

            # write far nodes in farNodeList
            farNodeList = []
            farNode = np.array(FarDefect[i])
            # far node Id
            farNodeList.append(len(node_list))
            # far node l'axe x
            farNodeList.append(farNode[0])
            # far node l'axe y
            farNodeList.append(farNode[1])
            # add far node in the node list
            node_list.append(farNodeList)
            # draw far node in frame
            cv2.circle(frame, FarDefect[i], 10, [100, 0, 255], 3)

            # write end nodes in endNodeList
            endNodeList = []
            endNode = np.array(EndDefect[i])
            # end node Id
            endNodeList.append(len(node_list))
            # end node l'axe x
            endNodeList.append(endNode[0])
            # end node l'axe y
            endNodeList.append(endNode[1])
            # add end node in the node list
            node_list.append(endNodeList)
            # draw end node in frame
            cv2.circle(frame, EndDefect[i], 10, [255, 0, 100], 3)

            # Add the edge from start node to far node
            edge = []
            # edge Id
            edge.append(len(edge_list))
            # edge source : start node
            edge.append(startNodeList[0])
            # edge target : far node
            edge.append(farNodeList[0])
            # edge distance
            edge.append(distanceBetweenDefectsToStart[i])
            # add edge in the edge list
            edge_list.append(edge)
            # cv2.line(frame, StartDefect[i], FarDefect[i], [255, 0, 100], 2)

            # Add the edge from far node to end node
            edge = []
            # edge Id
            edge.append(len(edge_list))
            # edge source : far node
            edge.append(farNodeList[0])
            # edge target : end node
            edge.append(endNodeList[0])
            # edge distance
            edge.append(distanceBetweenDefectsToEnd[i])
            # add edge in the edge list
            edge_list.append(edge)
            # cv2.line(frame, FarDefect[i], EndDefect[i], [255, 0, 100], 2)

    # write nodes and edges in a list
    graph = []
    graph.append(node_list)
    graph.append(edge_list)
    # print graph
    return graph

# 7. Write in a file
def writeFile(graph):
    file = open('./graph/graph.txt', 'w')
    for i in graph:
        info = ' '.join([str(j) for j in i])
        file.write(info + "\n")
    file.close()


############################### Learning ###############################

def learning(fileName):
    # Read images
    frame = readImage(fileName)
    frame_original = np.copy(frame)

    # Filters and threshold
    frame_threshold = hand_threshold(frame)
    contour_frame = np.copy(frame_threshold)
    # print filter image
    cv2.imwrite('./result/frame_threshold.png', frame_threshold)

    # Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(contour_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found, hand_contour, StartDefect, EndDefect, FarDefect = hand_contour_find(contours, contour_frame)

    FarDefect, centerMass, moments = calculeCentralMass(hand_contour, FarDefect, contour_frame)
    # print hand contour image with Central Mass
    cv2.imwrite('./result/frame_CentralMass.png', contour_frame)
    # print distanceBetweenDefectsToCenter

    # Calcule distance between far defects to start nodes and end nodes
    distanceBetweenDefectsToStart,distanceBetweenDefectsToEnd, distanceBetweenEndNodeToNextStartNode = calculeDistance(StartDefect, EndDefect, FarDefect,centerMass)
    # Create a graph
    graph = creatGraph(StartDefect, EndDefect, FarDefect, contour_frame,distanceBetweenDefectsToStart, distanceBetweenDefectsToEnd, distanceBetweenEndNodeToNextStartNode)
    # Write in a file
    writeFile(graph)
    # print hand contour image after delete too small distance
    cv2.imwrite('./graph/frame_delete.png', contour_frame)

    # use graph image for calucalting 7 vertor using HuMoments
    vector_graph = cv2.HuMoments(cv2.moments(contour_frame)).flatten()
    # print vector_graph

    return vector_graph

############################### Recognitiom ###############################

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        numOflabel = labels[sortedDistances[i]]
        classCount[numOflabel] = classCount.get(numOflabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group=zeros((20,7))
    labels = ['Lizard', 'Lizard', 'Spock', 'Spock', 'Paper', 'Paper', 'Paper', 'Rock', 'Rock', 'Scissors', 'Scissors', 'Scissors', 'Scissors', 'Rock', 'Rock', 'Paper','Lizard', 'Lizard','Spock', 'Spock']

    for i in range(20):
        fileName = './learning/'+str(i)+'.jpeg'
        vector_graph = learning(fileName)
        for j in range(7):
            group[i,j]=vector_graph[j]

    fileNameTest = './testing/4.jpeg'
    vector_graph_test = learning(fileNameTest)

    my = classify(vector_graph_test, group, labels, 1)
    print(my)

