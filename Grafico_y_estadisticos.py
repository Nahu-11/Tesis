#Importo librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Variables que voy a usar
columnas = ["Momento","TSS1","TSS2","TS1","TS2","TS3","TS4","TS_QG","U","DIR",
"TA1","HR1","TA2","HR2","TA3","HR3","TA4","HR4","QG1","QG2","WET","RG","PAR",
"TSH1","HS1","GH1","Real(KH)1","Imag(KH)1","Real_T(KH)1","Imag_T(KH)1","TSH2",
"HS2","GH2","Real(KH)2","Imag(KH)2","Real_T(KH)2","Imag_T(KH)2","TSH3","HS3",
"GH3","Real(KH)3","Imag(KH)3","Real_T(KH)3","Imag_T(KH)3","TSH4","HS4","GH4",
"Real(KH)4","Imag(KH)4","Real_T(KH)4","Imag_T(KH)4","TSH5","HS5","GH5","Real(KH)5",
"Imag(KH)5","Real_T(KH)5","Imag_T(KH)5"]
cant_col = len(columnas)

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
media = list(range(cant_dias - 1))
varianza = list(range(cant_dias - 1))
asimetria = list(range(cant_dias - 1))
curtosis = list(range(cant_dias - 1))

#Separo los datos por variable y luego por mes
j=1
for y in columnas:
	if j < cant_col:
		
		#Creo directorios para guardar los gráficos y tablas según variable
		direccion = 'C:\\Users\\nahue\\Desktop\\Tesis\\Resultados\\' + columnas[j]
		if not os.path.exists(direccion):
			os.makedirs(direccion)
		
		maximo = data[columnas[j]].max()
		minimo = data[columnas[j]].min()
		#Termino de separar los datos por mes
		i=0
		for x in dias:
			if i < cant_dias - 1:
				inicial = dias[i]
				final = dias[i+1] - 1
				variable = data[columnas[j]].loc[inicial:final]
				
				#Calculo estadisticos
				mes[i]= str(inicial) + '-' + str(final)
				total[i] = variable.count()
				faltantes[i] = 96 * (final - inicial) - total[i]
				media[i] = variable.mean()
				varianza[i] = variable.var()
				asimetria[i] = variable.skew()
				curtosis[i] = variable.kurtosis()
				
				#Grafico para cada variable según el mes
				variable.plot(
					x ='Día',
					y = columnas[j],
					figsize = (18, 9),
					title = 'Días ' + mes[i],
					xlim = (inicial, final),
					ylim = (minimo, maximo),
					style = '.').set(
						xlabel = "Momento Juliano",
						ylabel = columnas[j]) 
				
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
			'Media': media,
			'Varianza': varianza,
			'Asimetría':asimetria,
			'Curtosis': curtosis}
		tabla_final = pd.DataFrame(data = tabla,)
		
		#Guardo la tabla
		tabla_final.to_csv(path_or_buf = direccion + '\\Estadisticos.csv', index = False)
	j = j + 1

del data
del tabla_final

#Este es un comentario inútil
# print('El script se ha ejecutado con éxito, presione una tecla para salir')
# input()

#Ejecuta el script para la Precipitacion
exec(open('C:\\Users\\nahue\\Desktop\\Tesis\\Precipitacion.py').read())