#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Reading the training data in a csv file
dbTraining = []
X = []
Y = []

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            dbTraining.append(row)

# Function to transform categorical data to numbers
def transform_data(data):
    transformed_data = []
    for row in data:
        transformed_row = []
        for col in row:
            if col == 'Sunny':
                transformed_row.append(1)
            elif col == 'Overcast':
                transformed_row.append(2)
            elif col == 'Rain':
                transformed_row.append(3)
            elif col == 'Hot':
                transformed_row.append(1)
            elif col == 'Mild':
                transformed_row.append(2)
            elif col == 'Cool':
                transformed_row.append(3)
            elif col == 'High':
                transformed_row.append(1)
            elif col == 'Normal':
                transformed_row.append(2)
            elif col == 'Yes':
                transformed_row.append(1)
            elif col == 'No':
                transformed_row.append(2)
        transformed_data.append(transformed_row)
    return transformed_data

X = transform_data([row[:-1] for row in dbTraining])
Y = [1 if row[-1] == 'Yes' else 2 for row in dbTraining]

clf = GaussianNB()
clf.fit(X, Y)

dbTest = []

with open('weather_test.csv', 'r') as testfile:
    testreader = csv.reader(testfile)
    for j, test_row in enumerate(testreader):
        if j > 0:  # Skipping the header
            dbTest.append(test_row)

print("Day\tOutlook\tTemperature\tHumidity\tWind\tPlayTennis\tConfidence")

for test_instance in dbTest:
    transformed_test_data = transform_data([test_instance])
    prob_predictions = clf.predict_proba(transformed_test_data)[0]

    if max(prob_predictions) >= 0.75:
        prediction = 'Yes' if clf.predict([transformed_test_data[0]])[0] == 1 else 'No'
        confidence = max(prob_predictions)
        print(f"{test_instance[0]}\t{test_instance[1]}\t{test_instance[2]}\t\t{test_instance[3]}\t{test_instance[4]}\t\t\t\t{prediction}\t\t{confidence:.2f}")
