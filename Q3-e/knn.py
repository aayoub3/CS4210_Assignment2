#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: # SKIPPING THE HEADER
         db.append(row)

total_misclassifications = 0

for test_index, test_instance in enumerate(db):

    X = []
    Y = []

    for train_index, train_instance in enumerate(db):
        if test_index != train_index:
            X.append([float(train_instance[0]), float(train_instance[1])])
            Y.append(train_instance[2])

    testSample = [float(test_instance[0]), float(test_instance[1])]

    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    class_predicted = clf.predict([testSample])[0]

    true_label = test_instance[2]

    if class_predicted != true_label:
        total_misclassifications += 1

error_rate = total_misclassifications / len(db)

print(f'LOO-CV Error Rate for 1NN: {error_rate:.2f}')






