import pandas as pd 
import math

data = pd.read_csv("diabetes.csv")

diabetes = data[data["Outcome"] == 1]
non_diabetes = data[data["Outcome"] == 0]

diabetes_list = []
non_diabetes_list = []
k_list = []
data_list = []

for index, row in data.iterrows():
    age = row['Age']
    glucose = row['Glucose']
    outcome = row['Outcome']
    data_list.append((age,glucose,outcome))
    
    if outcome == 1:
        diabetes_list.append((age, glucose))
    else:
        non_diabetes_list.append((age, glucose))

age = input("Age:")
glucose = input("glucose:")
kValue = int(input("k Value:"))

for data in data_list:
    euclideanDistance = math.sqrt((float(age)-data[0])**2 + ((float(glucose)-data[1])**2))
    k_list.append((data,euclideanDistance))

sorted_k_list = sorted(k_list,key=lambda x: x[1])

print(sorted_k_list)

k_nearest_neighbors = [item[0] for item in sorted_k_list[:kValue]]

print(k_nearest_neighbors)
diabetesCount = 0
nonDiabetesCount = 0

for data in k_nearest_neighbors:
    if(data[2]==1):
        diabetesCount += 1
    else:
        nonDiabetesCount += 1

print( diabetesCount)
print(nonDiabetesCount)

if(diabetesCount>nonDiabetesCount):
    print("diabetic")
else:
    print("not diabetic")
