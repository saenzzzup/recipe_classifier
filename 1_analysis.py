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
For instance, as we can see in the first plot, Italian and Mexican receipts represent more than
a third of the entire dataset. So, it is probable that this will affect how our vectors form. 
It is good to keep this on mind for this or any other further model we apply to this dataset. 

Let’s check if there is a bias on the size of the receipts.
"""
num_ingredients = dict(zip(counts_cuis.keys(), [list() for x in counts_cuis.keys()]))
print('='*15,'\ncuisine \tmean')
for cu in counts_cuis.keys():
  num_ingredients[cu] = [
    len(x) for x in trainData[trainData['cuisine'] == cu]['ingredients']]

for cu, frq in num_ingredients.items():
  print(f'{cu}    \t{np.mean(frq):.2f}')
"""
The frequency of the ingredients presents a similar scenario, a few ingredients are tremendously popular.
Make sense, some ingredients as salt, or water are common in any recipes. Half of the ingredients only
appear 4 or less times in the dataset, that is wide less what I expected. 

Let's check the most popular.
"""
# This is to big to plot, let's check the percentiles
print('Mediana ingredientes: ', np.median(list(counts_ingr.values())))
print('Percentiles ingredientes: ', np.percentile(list(counts_ingr.values()), [25., 50., 75., 99.]))
"""
Half of the ingredients only appear 4 or less times in the dataset, that is wide
less what I expected. 

Let's check the most populars.
"""
# top 15
print('top 15 ingredientes: ', counts_ingr.most_common(15))
"""
A few ingredients like Salt and water make a lot of sense that they are highly frequent,
but the present of olive oil among these omnipresent ingredients make me think that is an artefact
of the bias of the dataset to the Italian cooking.
"""
# Tail 50
print('10 ingredientes menos populares: ', counts_ingr.most_common()[-15:])
"""
There are some very specific ingredients... I expect that some of those are typos, or just versions
of other ingredients. Also notice that in the dataset the same ingredient can present in different 
formats, garlic, and garlic cloves. 

First a quick search for parenthesis or similar symbols that rise a red flag to typos, or weird writing
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
Well, I guess some pre-processing could be good, but let's see how our model behave. 
Let's train the neural network with a raw version of the dataset
"""
print('Simbolos mas populares: ', counts_symbols.most_common(20))
