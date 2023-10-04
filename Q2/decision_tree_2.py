#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# transform data to numeric
def transform_data(data):
    transformed_data = []
    for row in data:
        transformed_row = []
        for col in row[:-1]:
            if col == 'Young':
                transformed_row.append(1)
            elif col == 'Prepresbyopic':
                transformed_row.append(2)
            elif col == 'Presbyopic':
                transformed_row.append(3)
            elif col == 'Myope':
                transformed_row.append(1)
            elif col == 'Hypermetrope':
                transformed_row.append(2)
            elif col == 'Yes':
                transformed_row.append(1)
            elif col == 'No':
                transformed_row.append(2)
        transformed_data.append(transformed_row)
    return transformed_data

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    X = transform_data(dbTraining)

    for row in dbTraining:
        if row[-1] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)

    total_accuracy = 0

    #loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        dbTest = []

        # read the test data and add this data to dbTest
        with open('contact_lens_test.csv', 'r') as testfile:
            testreader = csv.reader(testfile)
            for j, test_row in enumerate(testreader):
                if j > 0:  # SKIP THE HEADER
                    dbTest.append(test_row)

        correct_predictions = 0

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            transformed_test_data = transform_data([data[:-1]])

            class_predicted = clf.predict(transformed_test_data)[0]

            true_label = 1 if data[-1] == 'Yes' else 2

            if class_predicted == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(dbTest)
        total_accuracy += accuracy

    # find the average of this model during the 10 runs (training and test set)
    average_accuracy = total_accuracy / 10

    print(f'Final accuracy for {ds}: {average_accuracy:.2f}')


