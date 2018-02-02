import csv, random, math, operator

# Loads data from file into a training set and test set randomly
# based on a provided weight. 2/3 is recommended as training data.
def loaddata(filename, split):
    trainingset = []
    testset = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header

        for row in reader:
            # cast the 4 first columns to float
            for x in range(4):
                row[x] = float(row[x])

            if random.random() < split:
                trainingset.append(row)
            else:
                testset.append(row)

    return (trainingset, testset)

# Calculates the euclidian distance between two data instances (similarities),
# using the 4 first positions as feature attributes. These must be numeric
# and use the same units, in this case, centimeters.
# Eucledian distance is defined as:
# the square root of the sum of the squared differences between the two
# instances
def euclidean_distance(instance1, instance2, feature_attributes = 4):
    distance = 0

    for x in range(feature_attributes):
        distance += pow(instance1[x] - instance2[x], 2)

    return math.sqrt(distance)

# Gets the nearest instances from training set
# k determines how far we look from testinstance
def get_neighbors(trainingset, testinstance, k):
    distances = []

    for traininginstance in trainingset:
        distance = euclidean_distance(
                testinstance,
                traininginstance,
                len(testinstance) - 1)
        distances.append((traininginstance, distance))

    distances.sort(key = operator.itemgetter(1)) # sort by 2nd entry in tuple

    # at this point, we have all entire training set with a similarity value
    # (distance) on how similar the testinstance is to each of every traning
    # value. We want to fine-grain this by using the k value as a 'limit'.
    # We are returning a plain list of neighbor instances, wihout distances.
    return [measurement[0] for measurement in distances[0:k]]

# Calculates the total score for each class of the neighbors. We do this based
# on the total number of neighbors of the same class.
def get_response(neighbors):
    class_votes = {}

    for neighbor in neighbors:
        class_name = neighbor[-1]

        if class_name in class_votes:
            class_votes[class_name] += 1
        else:
            class_votes[class_name] = 0

    # sort the best match based on the vote
    return sorted(class_votes.items(), key=operator.itemgetter(1))[0][0]

# Checks the outcoming predictions agains the actual values in the
# test data set, to see how accurate our preduction was with kNN
def get_accuracy(testset, predictions):
    correct = 0

    for x in range(len(testset)):
        correct += 1 if testset[x][-1] == predictions[x] else 0

    return (correct/float(len(testset))) * 100.0

# Main program
if __name__ == '__main__':
    # Load 2/3 into training set from file
    (trainingset, testset) = loaddata('iris.csv', 0.66)
    predictions = []
    k = 3

    print('No. training sets: ' + str(len(trainingset)))
    print('No. test sets: ' + str(len(testset)))

    for testinstance in testset:
        neighbors = get_neighbors(trainingset, testinstance, 3)
        actual = testinstance[-1]
        predicted = get_response(neighbors) # get the winning class name

        predictions.append(predicted)

        print('Predicted: ' + predicted + ', actual: ' + actual)

    accuracy = get_accuracy(testset, predictions)

    print('Accuracy of test data: ' + repr(accuracy) + '%')
