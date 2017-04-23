from sklearn import svm, cross_validation, tree
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import csv

def add_to_database(row,X,Y):

	# Remove the first three elements and convert rest to int
	row = row[3:]
	row = [float(i) for i in row]

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
	with open('sent_embed_avg.csv','r') as csvfile:
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

#################################################################################

# #################################
# ############## SVM ##############

# #########  NORMAL  SVM  #########

# print "Normal SVM:\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = svm.SVC(kernel='linear')
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


# ############ K-FOLD #############

# print "\n\nK-FOLD SVM:\n-----------------\n"

# clf_kfold = svm.SVC(kernel='linear')
# scores = cross_val_score(clf_kfold, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


####################################################################################

# #################################
# ######## Decision Trees #########

# #########  NORMAL  DT  ##########

# print "Normal DT:\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = tree.DecisionTreeClassifier()
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


# ############ K-FOLD #############

# print "\n\nK-FOLD DT:\n-----------------\n"

# clf_kfold = tree.DecisionTreeClassifier()
# scores = cross_val_score(clf_kfold, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

# #################################
# ######## AdaBoost #########

# #########  NORMAL  Adaboost  ##########

# for i in range(10):

# 	print "Normal Adaboost (i = " + str(i+1) + ") :\n-----------------\n"

# 	# APPLY SVM and specify kernel
# 	clf = AdaBoostClassifier(n_estimators=10*(i+1))
# 	model = clf.fit(trainX, trainY)

# 	# Train Accuracy
# 	trainPrediction = clf.predict(trainX)
# 	trainError = trainPrediction - trainY
# 	trainCorrect = sum(x == 0 for x in trainError)
# 	trainAccuracy = float(trainCorrect)/len(trainY) * 100
# 	print "Training Accuracy = " + str(trainAccuracy)

# 	# Test Accuracy
# 	testPrediction = clf.predict(testX)
# 	testError = testPrediction - testY
# 	testCorrect = sum(x == 0 for x in testError)
# 	testAccuracy = float(testCorrect)/len(testY) * 100
# 	print "Testing Accuracy = " + str(testAccuracy)


# 	############ K-FOLD #############

# 	print "\n\nK-FOLD Adaboost (i = " + str(i+1) + ") :\n-----------------\n"

# 	clf_kfold = AdaBoostClassifier(n_estimators=10*(i+1))
# 	scores = cross_val_score(clf_kfold, X, Y, cv=10)

# 	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

#################################
######## GradientBoosting #########

#########  NORMAL  Gradboost  ##########

# max_score = 0
# params = [10,0.1,1]

# for i in range(4):
# 	for lr in range(5):
# 		for md in range(2):

# 			print "Normal Gradboost (i = " + str(i+1) + ") :\n-----------------\n"

# 			# APPLY SVM and specify kernel
# 			clf = GradientBoostingClassifier(n_estimators=10*(i+1))
# 			model = clf.fit(trainX, trainY)

# 			# Train Accuracy
# 			trainPrediction = clf.predict(trainX)
# 			trainError = trainPrediction - trainY
# 			trainCorrect = sum(x == 0 for x in trainError)
# 			trainAccuracy = float(trainCorrect)/len(trainY) * 100
# 			print "Training Accuracy = " + str(trainAccuracy)

# 			# Test Accuracy
# 			testPrediction = clf.predict(testX)
# 			testError = testPrediction - testY
# 			testCorrect = sum(x == 0 for x in testError)
# 			testAccuracy = float(testCorrect)/len(testY) * 100
# 			print "Testing Accuracy = " + str(testAccuracy)


# 			########### K-FOLD #############

# 			print "\n\nK-FOLD Gradboost (i = " + str(i+1) + ") :\n-----------------\n"

# 			new_params = [10*(i+7),float( float(2*lr+2)/10),2]
# 			print new_params

# 			clf_kfold = GradientBoostingClassifier(n_estimators=new_params[0], learning_rate=new_params[1], max_depth=new_params[2], random_state=0)
# 			scores = cross_val_score(clf_kfold, X, Y, cv=10)

# 			if scores.mean() > max_score:
# 				max_score = scores.mean()
# 				params = new_params

# 			print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# print "\n\n-----------------------\n\n"
# print "Accuracy: %0.4f" % max_score
# print "Parameters:"
# print params

####################################################################################

# #################################
# ######## Logistic Regression #########

# #########  NORMAL  LR  ##########

# print "Normal LR:\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = LogisticRegression(random_state=1)
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


# ############ K-FOLD #############

# print "\n\nK-FOLD LR:\n-----------------\n"

# clf_kfold = LogisticRegression(random_state=1)
# scores = cross_val_score(clf_kfold, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

# #################################
# ######## Gaussin NB #########

# #########  NORMAL  GNB  ##########

# print "Normal GNB:\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = GaussianNB()
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


# ############ K-FOLD #############

# print "\n\nK-FOLD GNB:\n-----------------\n"

# clf_kfold = GaussianNB()
# scores = cross_val_score(clf_kfold, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

# #################################
# ######## Random Forest #########

# #########  NORMAL  RF  ##########

# for i in range(20):

# 	# print "Normal RF (i = " + str(i+1) + ") :\n-----------------\n"

# 	# # APPLY SVM and specify kernel
# 	# clf = RandomForestClassifier(n_estimators=i+1)
# 	# model = clf.fit(trainX, trainY)

# 	# # Train Accuracy
# 	# trainPrediction = clf.predict(trainX)
# 	# trainError = trainPrediction - trainY
# 	# trainCorrect = sum(x == 0 for x in trainError)
# 	# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# 	# print "Training Accuracy = " + str(trainAccuracy)

# 	# # Test Accuracy
# 	# testPrediction = clf.predict(testX)
# 	# testError = testPrediction - testY
# 	# testCorrect = sum(x == 0 for x in testError)
# 	# testAccuracy = float(testCorrect)/len(testY) * 100
# 	# print "Testing Accuracy = " + str(testAccuracy)


# 	############ K-FOLD #############

# 	print "\n\nK-FOLD RF (i = " + str(i+1) + ") :\n-----------------\n"

# 	clf_kfold = RandomForestClassifier(n_estimators=10*(i+1))
# 	scores = cross_val_score(clf_kfold, X, Y, cv=10)

# 	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

# #################################
# ######## Extremely Random Trees #########

# #########  NORMAL  RF  ##########

# for i in range(20):

# 	# print "Normal RF (i = " + str(i+1) + ") :\n-----------------\n"

# 	# # APPLY SVM and specify kernel
# 	# clf = ExtraTreesClassifier(n_estimators=i+1)
# 	# model = clf.fit(trainX, trainY)

# 	# # Train Accuracy
# 	# trainPrediction = clf.predict(trainX)
# 	# trainError = trainPrediction - trainY
# 	# trainCorrect = sum(x == 0 for x in trainError)
# 	# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# 	# print "Training Accuracy = " + str(trainAccuracy)

# 	# # Test Accuracy
# 	# testPrediction = clf.predict(testX)
# 	# testError = testPrediction - testY
# 	# testCorrect = sum(x == 0 for x in testError)
# 	# testAccuracy = float(testCorrect)/len(testY) * 100
# 	# print "Testing Accuracy = " + str(testAccuracy)


# 	############ K-FOLD #############

# 	print "\n\nK-FOLD RF (i = " + str(i+1) + ") :\n-----------------\n"

# 	clf_kfold = ExtraTreesClassifier(n_estimators=10*(i+1))
# 	scores = cross_val_score(clf_kfold, X, Y, cv=10)

# 	print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

####################################################################################

#################################
######## Bagging #########

#########  NORMAL  bagging  ##########

# print "Normal RF (i = " + str(i+1) + ") :\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


# ############ K-FOLD #############

# print "\n\nK-FOLD Bagging:\n-----------------\n"

# clf_kfold = BaggingClassifier(RandomForestClassifier(n_estimators=100),max_samples=0.5, max_features=0.5)
# scores = cross_val_score(clf_kfold, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#####################################################################################

# ###############################
# ######### Combination #########

# clf_SVM = svm.SVC(kernel='linear')
# clf_DT = tree.DecisionTreeClassifier()
# clf_AdaBoost = AdaBoostClassifier(n_estimators=50)
# clf_GradBoost = GradientBoostingClassifier(n_estimators=80, learning_rate=0.6, max_depth=2, random_state=0)
# clf_LR = LogisticRegression(random_state=1)
# clf_RF = RandomForestClassifier(n_estimators=100)
# clf_ERT = ExtraTreesClassifier(n_estimators=110)

# print "\n\nK-FOLD Voting:\n-----------------\n"

# # eclf = VotingClassifier(estimators=[('dt', clf_DT), ('adaboost', clf_AdaBoost), ('gradBoost', clf_GradBoost), ('lr', clf_LR), ('rf', clf_RF), ('ert', clf_ERT)], voting='soft', weights=[2,4,4,3,5,5])
# eclf = VotingClassifier(estimators=[('adaboost', clf_AdaBoost), ('gradBoost', clf_GradBoost), ('rf', clf_RF), ('ert', clf_ERT)], voting='hard')
# scores = cross_val_score(eclf, X, Y, cv=10)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

#####################################################################################

# #################################
# ############## NN ##############

# #########  NORMAL  NN  #########

# print "Normal NN:\n-----------------\n"

# # APPLY SVM and specify kernel
# clf = svm.SVC(kernel='linear')
# model = clf.fit(trainX, trainY)

# # Train Accuracy
# trainPrediction = clf.predict(trainX)
# trainError = trainPrediction - trainY
# trainCorrect = sum(x == 0 for x in trainError)
# trainAccuracy = float(trainCorrect)/len(trainY) * 100
# print "Training Accuracy = " + str(trainAccuracy)

# # Test Accuracy
# testPrediction = clf.predict(testX)
# testError = testPrediction - testY
# testCorrect = sum(x == 0 for x in testError)
# testAccuracy = float(testCorrect)/len(testY) * 100
# print "Testing Accuracy = " + str(testAccuracy)


############ K-FOLD #############

print "\n\nK-FOLD NN:\n-----------------\n"

clf_kfold = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
scores = cross_val_score(clf_kfold, X, Y, cv=10)

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

