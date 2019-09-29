#Importo librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

#Variables que voy a usar
columnas = ["Momento", "PP"]

#Cargo los datos
data = pd.read_csv(
    'C:\\Users\\nahue\\Desktop\\Tesis\\analisis.csv',
    delimiter = ';',
    decimal = '.',
    skiprows = [1,2],
    index_col = 0,
    usecols = columnas,
    dtype = 'float64',
    engine = 'python'
    )

#Defino variables que voy a usar
dias = [32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
cant_dias = len(dias)
mes = list(range(cant_dias - 1))
total = list(range(cant_dias - 1))
faltantes = list(range(cant_dias - 1))
acumulada = list(range(cant_dias - 1))
columnas = str(columnas[1])

#Creo directorios para guardar los gráficos y tablas según variable
direccion = 'C:\\Users\\nahue\\Desktop\\Tesis\\Resultados\\' + columnas
if not os.path.exists(direccion):
	os.makedirs(direccion)
		
maximo = data[columnas].max()

#Separo los datos por mes
i=0
for x in dias:
	if i < cant_dias - 1:
		inicial = dias[i]
		final = dias[i+1] - 1
		variable = data.loc[inicial:final]
				
		#Calculo estadisticos
		mes[i]= str(inicial) + '-' + str(final)
		total[i] = variable.count()
		faltantes[i] = 96 * (final - inicial) - total[i]
		acumulada[i] = variable.sum()
				
		#Grafico para cada variable según el mes
		variable.plot(
			y = columnas,
			figsize = (18, 9),
			title = 'Días ' + mes[i],
			xlim = (inicial, final),
			ylim = (0, maximo),
			style = '.').set(
				xlabel = "Momento Juliano",
				ylabel = columnas) 
		
		del variable
		
		#Guardo los gráficos
		k = str(i + 2)
		plt.savefig(direccion + '\\' + k + ' Dias ' + mes[i] + '.png')
		plt.close()
	i = i + 1

#Creo una tabla de estadisticos para cada mes
tabla = {
	'Dias': mes,
	'Datos utilizados' : total,
	'Datos faltantes' : faltantes,
	'Precipitación Acumulada': acumulada}
tabla_final = pd.DataFrame(data = tabla,)
		
#Guardo la tabla
tabla_final.to_csv(path_or_buf = direccion + '\\Estadisticos.csv', index = False)

del data
del tabla_final

print('El script se ha ejecutado con éxito, presione una tecla para salir')
input()