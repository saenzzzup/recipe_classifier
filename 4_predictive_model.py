import json
import time
import pandas as pd
import numpy as np
import sklearn
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
"""# Carga de datos y modelo
Se cargan los datos de entrenamiento y prueba, junto con los datos generadas previamente con el modelo Word2Vec
"""

trainData = pd.read_json ("./data/train.json")
testData = pd.read_json("./data/test.json")

model_name = "./features_word2vec"
model = Word2Vec.load(model_name)
index2word_set = set(model.wv.index2word)

"""# Construcción de particiones
Se construyen los datos de entrada a partir de las recetas en cada set. Para cada ingrediente de cada receta, se extrae su vector de características generado en Word2Vec,  generando una matriz de características para cada set.
"""
X = trainData['ingredients']
Y = trainData['cuisine']
try:
  with open('train-tes.data', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)
  print('Test and train data loaded')
except:
  print('Partitioning data')
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
  with open('train-tes.data', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
try:
  with open('f_matrixes', 'rb') as f:
    f_matrix_train, f_matrix_test = pickle.dump(f)
except:
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
      recipeFeatureVecs[counter] = makeFeatureVec(recipes.iloc[i], model, num_features)
      counter = counter + 1
    return recipeFeatureVecs

  print('Building ingredients vectors..')
  num_features = 300
  f_matrix_train = getAvgFeatureVecs(x_train, model, num_features)
  f_matrix_test = getAvgFeatureVecs(x_test, model, num_features)
  
  with open('f_matrixes', 'wb') as f:
    pickle.dump((f_matrix_train, f_matrix_test), f)

"""# Random Forest
Contruye y entrena el modelo de Random Forest para clasificar las recetas por sus ingredientes. Se utilizaron 100 arboles de decisión dentro del modelo.

### Entrenamiento de modelo
"""
try:
  with open('forest.ai', 'rb') as file:
    forest = pickle.load(file)
  print('Forest model loaded')
except:
  # Fitting a random forest classifier to the training data
  forest = RandomForestClassifier(n_estimators = 600)

  print("Fitting random forest to training data....")
  t0 = time.time()    
  forest.fit(f_matrix_train, y_train)
  t1 = time.time()
  print(f'Tiempo de ejecucion: {t1-t0}')

  with open('forest.ai', 'wb') as file:
    pickle.dump((forest), file)
  """### Prueba de modelo"""

  # Predicting the sentiment values for test data and saving the results in a csv file 
  """ t0 = time.time()
  resultForest = forest.predict(f_matrix_test)
  t1 = time.time() """
  #outputForest = pd.DataFrame(data={"id":testData["id"], "cuisine":resultForest})
  #print(f'Tiempo de ejecucion: {t1-t0}')

  """# MLP
  Contruye y entrena el modelo de MLP para clasificar las recetas por sus ingredientes. Se utiliza una configuración de 3 capas con 100, 60 y 20 neuronas respectivamente.

  ### Entrenamiento de modelo
  """
try:
  with open('mlp.ai', 'rb') as file:
    m = pickle.load(file)
  print('MLP model loaded')
except:
  m = MLPClassifier(solver='adam', hidden_layer_sizes=(100,60,20), random_state=1)
  print("Fitting MLP to training data....") 
  t0 = time.time()   
  m.fit(f_matrix_train, y_train)
  t1 = time.time()
  print(f'Tiempo de ejecucion: {t1-t0}')

  with open('mlp.ai', 'wb') as file:
    pickle.dump((m), file)
  """### Prueba de modelo"""

  # Predicting the sentiment values for test data and saving the results in a csv file 
  """ t0 = time.time()
  resultMLP = m.predict(f_matrix_test)
  t1 = time.time() """
  #outputMLP = pd.DataFrame(data={"id":testData["id"], "cuisine":resultMLP})
  #print(f'Tiempo de ejecucion: {t1-t0}')

"""# Evaluación
Pruebas de evaluación entre ambos modelos (Random Forest y MLP) de acuerdo al porcentaje de precisión.
"""
""" 
comparison = pd.DataFrame(data={"id":testData["id"], "Cuisine MLP":resultMLP, "Cuisine Forest":resultForest})
comparison.head(10) """

print('MLP: ',  m.score(f_matrix_test, y_test))
print('RF: ', forest.score(f_matrix_test, y_test))
"""
output = pd.DataFrame(data={"id":trainData["id"], "Cuisine":trainData['cuisine'], "Cuisine MLP":m.predict(f_matrix_train), "Cuisine Forest":forest.predict(f_matrix_train)})

accMLP = output[output['Cuisine MLP'] == output['Cuisine']]['id'].count()/len(output)
accForest = output[output['Cuisine Forest'] == output['Cuisine']]['id'].count()/len(output)

print(f'Accuracy: {accMLP}')
print(f'Accuracy: {accForest}')
"""