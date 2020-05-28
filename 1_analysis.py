# Manejo de datos
import json 
import operator
import collections
import re

# Manejo de datos en estructura de tablas
import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Información de entrenamiento del modelo
trainData = pd.read_json(open("./data/train.json", 'r'))

# Extraer ingredientes de las recetas (Eliminar cantidades, medidas, etc.)
raw_ingredients = list()
for ingredients in trainData['ingredients']:
  raw_ingredients.extend([re.sub(r'\(.*oz.\)|\%|0-9|crushed|crumbles|ground|minced|powder|powdered|fresh|chopped|sliced', '', ingr).strip() for ingr in ingredients])

# Extraer los tipos de cocina
raw_cuisines = list()
for cuisine in trainData['cuisine']:
  raw_cuisines.append(cuisine.strip())

# Conteo de la frecuencia en los tipos de cocina
# return dictionary key:value (value:frecuency)
counts_ingr = collections.Counter(raw_ingredients)
counts_cuis = collections.Counter(raw_cuisines)
# Muestra de como se ve la información de los ingredientes.
print('Ingredientes totales (repetidos):  \t{}'.format((len(raw_ingredients))))
print('Ingredientes totales (no repetidos): \t{}'.format((len(counts_ingr.values()))))

# Esto proporcionará una distribución de cocinas,
# información indirecta de los ingredientes.
print('Numero total de recetas \t\t{}'.format(len(raw_cuisines)))
print('Numero total de cocinas \t\t{}'.format((len(counts_cuis.values()))))

# top 10
counts_cuis.most_common(10)

# Plot de la información
x_cu = [cu for cu, frq in counts_cuis.most_common()]
y_frq = [frq for cu, frq in counts_cuis.most_common()]
fbar = sns.barplot(x=x_cu, y=y_frq)
plt.show()
# xlabels
for item in fbar.get_xticklabels():
  item.set_rotation(90)
"""
Revisamos si existe una desviación en los ingredientes
"""
num_ingredients = dict(zip(counts_cuis.keys(), [list() for x in counts_cuis.keys()]))
print('='*15,'\ncuisine \tmean')
for cu in counts_cuis.keys():
  num_ingredients[cu] = [
    len(x) for x in trainData[trainData['cuisine'] == cu]['ingredients']]

for cu, frq in num_ingredients.items():
  print(f'{cu}    \t{np.mean(frq):.2f}')
"""
Buscamos los ingredientes más populares. Tiene sentido que algunos
"""
# This is to big to plot, let's check the percentiles
print('Mediana ingredientes: ', np.median(list(counts_ingr.values())))
print('Percentiles ingredientes: ', np.percentile(list(counts_ingr.values()), [25., 50., 75., 99.]))
"""
Algunos ingredientes sólo aparecen muy pocas veces (solo 4) en la base de datos.
"""
# top 15
print('top 15 ingredientes: ', counts_ingr.most_common(15))
"""
Imprimimos los ingredientes mas populares y tambien los menos populares.
"""
# Tail 50
print('10 ingredientes menos populares: ', counts_ingr.most_common()[-15:])
"""
Buscamos si existen signos especiales, mayusculas, minusculas o caracteristicas especiales
"""
symbols = list()

for ingredients in trainData['ingredients']:
  # I want ingredient remove
  for ingredient in ingredients:
    if re.match("\(|@|\$\?", ingredient.lower()):
      symbols.append(ingredient)

#symbols
counts_symbols = collections.Counter(symbols)
"""
Imprimimos los simbolos mas utilizados, despues los quitaremos. 
"""
print('Simbolos mas populares: ', counts_symbols.most_common(20))
