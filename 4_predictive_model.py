# Manejo de datos
import json

# Manejo de datos en estructura de tablas
import pandas as pd
import numpy as np

import time

# Modelo
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

"""
Carga de datos y modelo
Se cargan los datos de entrenamiento y prueba, junto con los datos generadas previamente con el modelo Word2Vec
"""

trainData = pd.read_json ("data/train.json")
testData = pd.read_json("data/test.json")

model_name = "./features_word2vec"
model = Word2Vec.load(model_name)
index2word_set = set(model.wv.index2word)

"""
Construcción de particiones
Se construyen los datos de entrada a partir de las recetas en cada set. Para cada ingrediente de cada receta, se extrae su vector de características generado en Word2Vec,  generando una matriz de características para cada set.
"""

def makeFeatureVec(recipieIngridients, model, num_features):
   
  featureVec = np.zeros((num_features,), dtype="float32")
  recipieIngridientsAmount = 0
  
  # Set de palabras en el modelo 
  index2word_set = set(model.wv.index2word)
  #
  # Se recorre cada palabra de la receta y se sustituye 
  # con su vector de caracteristicas
  for word in recipieIngridients:
    if word in index2word_set: 
      recipieIngridientsAmount = recipieIngridientsAmount + 1
      featureVec = np.add(featureVec, model.wv[word])
  # Se dividen los resultados entre el numero de ingredientes
  if recipieIngridientsAmount == 0:
    recipieIngridientsAmount = 1
  featureVec = np.divide(featureVec, recipieIngridientsAmount)
  return featureVec

def getAvgFeatureVecs(recipes, model, num_features):
  # Construye la matriz de características para cada receta
  recipeFeatureVecs = np.zeros((len(recipes), num_features), dtype="float32")
  counter = 0
  # Loop de recetas
  for i in range(len(recipes)):
    # Obtiene vectores de ingredientes para cada receta
    recipeFeatureVecs[counter] = makeFeatureVec(recipes.iloc[i]['ingredients'], model, num_features)
    counter = counter + 1
  return recipeFeatureVecs

num_features = 300
f_matrix_train = getAvgFeatureVecs(trainData, model, num_features)
f_matrix_test = getAvgFeatureVecs(testData, model, num_features)

"""
Random Forest
Contruye y entrena el modelo de Random Forest para clasificar las recetas por sus ingredientes. Se utilizaron 100 arboles de decisión dentro del modelo.
"""

# ENTRENAMIENTO DEL MODELO RANDOM FOREST

# Llenado del random forest classifier para el entrenamiento de los datos.
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting random forest to training data....")
t0 = time.time()    
forest.fit(f_matrix_train, trainData["cuisine"])
t1 = time.time()
print(f'Tiempo de ejecución de entrenamiento de Random Forest: {t1-t0}')

# PRUEBA DE MODELO RANDOM FOREST

# Predicting the sentiment values for test data and saving the results in a csv file 
t0 = time.time()
resultForest = forest.predict(f_matrix_test)
t1 = time.time()
outputForest = pd.DataFrame(data={"id":testData["id"], "cuisine":resultForest})
print(f'Tiempo de ejecución Random Forest: {t1-t0}')

"""
MLP
Contruye y entrena el modelo de MLP para clasificar las recetas por sus ingredientes. Se utiliza una configuración de 3 capas con 100, 60 y 20 neuronas respectivamente.
"""

# ENTRENAMIENTO DEL MODELO MLP

m = MLPClassifier(solver='adam', hidden_layer_sizes=(100,60,20), random_state=1)
print("Fitting MLP to training data....") 
t0 = time.time()   
m.fit(f_matrix_train, trainData["cuisine"])
t1 = time.time()
print(f'Tiempo de ejecución de entrenamiento MLP: {t1-t0}')

# PRUEBA DE MODELO MLP

# Predicting the sentiment values for test data and saving the results in a csv file 
t0 = time.time()
resultMLP = m.predict(f_matrix_test)
t1 = time.time()
outputMLP = pd.DataFrame(data={"id":testData["id"], "cuisine":resultMLP})
print(f'Tiempo de ejecución de MLP: {t1-t0}')

"""
Evaluación
Pruebas de evaluación entre ambos modelos (Random Forest y MLP) de acuerdo al porcentaje de precisión.
"""

comparison = pd.DataFrame(data={"id":testData["id"], "Cuisine MLP":resultMLP, "Cuisine Forest":resultForest})
comparison.head(10)

m.score(f_matrix_train,trainData['cuisine'])

forest.score(f_matrix_train,trainData['cuisine'])

output = pd.DataFrame(data={"id":trainData["id"], "Cuisine":trainData['cuisine'], "Cuisine MLP":m.predict(f_matrix_train), "Cuisine Forest":forest.predict(f_matrix_train)})

accMLP = output[output['Cuisine MLP'] == output['Cuisine']]['id'].count()/len(output)
accForest = output[output['Cuisine Forest'] == output['Cuisine']]['id'].count()/len(output)

print(f'Accuracy MLP: {accMLP}')
print(f'Accuracy Random Forest: {accForest}')