#Importo librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Cargo los datos
data = pd.read_csv(
		'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Marchi_2018_2.csv',
		delimiter = ',',
		decimal = '.',
		index_col = 0,
		engine = 'python'
		)

i=3
while i < 13:
	data_2 = pd.read_csv(
		'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Marchi_2018_'+ str(i) +'.csv',
		delimiter = ',',
		decimal = '.',
		index_col = 0,
		engine = 'python'
		)
	data = data.append(data_2, sort = False)
	i = i + 1

#Arreglo los NaN y guardo en un .csv
data = data.replace(-9999.900391,np.NaN)
data = data.replace(-9999.9003906,np.NaN)
data.to_csv(path_or_buf = 'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Datos.csv')

print('El script se ha ejecutado con exito')
input()


