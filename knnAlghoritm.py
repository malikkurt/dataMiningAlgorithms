import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('diabetes.csv')

X = data[['Age', 'Glucose']]
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()

param_grid = {'n_neighbors': list(range(1, 21))}

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f'Best k-Value: {best_k}')
print(f'Best Accuracy: {best_score}')

new_data = [[40, 150]]
new_data_scaled = scaler.transform(new_data)

prediction = grid_search.predict(new_data_scaled)

if prediction[0] == 1:
    print('This patient have diabetes.')
else:
    print('This patient may not have diabetes.')


# import pandas as pd 
# import math

# data = pd.read_csv("diabetes.csv")

# diabetes = data[data["Outcome"] == 1]
# non_diabetes = data[data["Outcome"] == 0]

# diabetes_list = []
# non_diabetes_list = []
# k_list = []
# data_list = []

# for index, row in data.iterrows():
#     age = row['Age']
#     glucose = row['Glucose']
#     outcome = row['Outcome']
#     data_list.append((age,glucose,outcome))
    
#     if outcome == 1:
#         diabetes_list.append((age, glucose))
#     else:
#         non_diabetes_list.append((age, glucose))

# age = input("Age:")
# glucose = input("glucose:")
# kValue = int(input("k Value:"))

# for data in data_list:
#     euclideanDistance = math.sqrt((float(age)-data[0])**2 + ((float(glucose)-data[1])**2))
#     k_list.append((data,euclideanDistance))

# sorted_k_list = sorted(k_list,key=lambda x: x[1])

# print(sorted_k_list)

# k_nearest_neighbors = [item[0] for item in sorted_k_list[:kValue]]

# print(k_nearest_neighbors)
# diabetesCount = 0
# nonDiabetesCount = 0

# for data in k_nearest_neighbors:
#     if(data[2]==1):
#         diabetesCount += 1
#     else:
#         nonDiabetesCount += 1

# print( diabetesCount)
# print(nonDiabetesCount)

# if(diabetesCount>nonDiabetesCount):
#     print("diabetic")
# else:
#     print("not diabetic")

