from sklearn import svm, cross_validation
from sklearn.cross_validation import cross_val_score
import csv

def add_to_database(row,X,Y):

	# Remove the first three elements and convert rest to int
	row = row[3:]
	row = [int(i) for i in row]

	# Last element is sentiment and remaining are the features
	# Check if last element is 0 (Don't train on neutral)
	if (row[-1] == 1):
		Y.append(1)
		X.append(row[:-1])
	elif (row[-1] == -1):
		Y.append(-1)
		X.append(row[:-1])

def build_X_Y(X,Y):
	
	first = 1
	with open('final_feature.csv','r') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			if (first):
				first = 0
			else:
				add_to_database(row,X,Y)

#####################################
######### MAIN CODE STARTS ##########

X = []
Y = []
build_X_Y(X,Y)
# 450 CASES TOTAL

# Split between train and test
trainX = X[:400]
trainY = Y[:400]
testX = X[401:]
testY = Y[401:]

#########  NORMAL  SVM  #########

print "Normal SVM:\n-----------------\n"

# APPLY SVM
clf = svm.SVC()
model = clf.fit(trainX, trainY)

# Train Accuracy
trainPrediction = clf.predict(trainX)
trainError = trainPrediction - trainY
trainCorrect = sum(x == 0 for x in trainError)
trainAccuracy = float(trainCorrect)/len(trainY) * 100
print "Training Accuracy = " + str(trainAccuracy)

# Test Accuracy
testPrediction = clf.predict(testX)
testError = testPrediction - testY
testCorrect = sum(x == 0 for x in testError)
testAccuracy = float(testCorrect)/len(testY) * 100
print "Testing Accuracy = " + str(testAccuracy)


############ K-FOLD #############

print "\n\nK-FOLD:\n-----------------\n"

clf_kfold = svm.SVC()
scores = cross_val_score(clf_kfold, X, Y, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
