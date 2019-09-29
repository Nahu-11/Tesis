#Importo librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import glob

z=2.85 #Altura de las mediciones

#Cargo datos
camino = 'D:\\Mediciones\\Filtrados\\*'
direccion = glob.glob( camino + '\\ustar.csv') 	#Para calcular los otros espectros cambiar aca
for j in direccion:
	guardado = j[0:-9] + 'ustar\\' 				#Para calcular los otros espectros cambiar aca (el -9 tambien)
	if not os.path.exists(guardado):
		os.makedirs(guardado)
	data = pd.read_csv(
			j,
			delimiter = ',',
			decimal = '.',
			engine = 'python'
			)
	
	# Defino algunas variables
	long = data['Path']
	max = long.count()
	fecha = list(range(max))
	obukhov = list(range(max))
	u_medio = list(range(max))
	u_star = list(range(max))
	r_cuadrado = list(range(max))
	fiteo_a = list(range(max))
	fiteo_b = list(range(max))
	error_a = list(range(max))
	error_b = list(range(max))
	
	#Abro espectros
	i = 0
	for x in long:
		mes = x[10:12]
		nombre = x[0:22]
		fecha[i] = x[7:-4]
		espectro = pd.read_csv(
				'D:\\Mediciones\\work\\2018-' + mes + '\\' + x,
				delimiter = ',',
				decimal = '.',
				index_col = False,
				engine = 'python'
				)
		
		#Calculo los numeros pi
		obukhov[i] = data['1/L'].iloc[i]
		u_medio[i] = data['u[m/s]         '].iloc[i]
		u_star[i] = data['ustar[m/s]     '].iloc[i]
		frec = espectro[' f']
		intensidad = espectro['Su']
													#Para calcular los otros espectros cambiar aca (el fi_e)
		if obukhov[i] < 0:
			fi_e= 1 + 0.5 * abs(z*obukhov[i]) **(2/3)
		else:
			fi_e = (1 + 5 * abs(z*obukhov[i])) **(2/3)
		espectro['pi_1'] = frec * z / u_medio[i]
		espectro['pi_u'] = frec * intensidad / (u_star[i]**2 * fi_e) #fi_e ya está elevado a la dos tercios, por eso no lo hago
		
		#Ajuste de los datos
		def potencial(a,b,c):
			return b*a**c
		x_fit = espectro['pi_1'].loc[espectro['pi_1'] > 0.1*z/u_medio[i]]
		y_fit = espectro['pi_u'].loc[espectro['pi_1'] > 0.1*z/u_medio[i]]
		fiteo, covarianzas = curve_fit(potencial, x_fit, y_fit)
		error = np.sqrt(np.diag(covarianzas))
		ajuste = potencial(x_fit,*fiteo)
		residuo = y_fit - ajuste
		ss_res = np.sum(residuo**2)
		ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
		r_cuadrado[i] = 1 - (ss_res / ss_tot)
		fiteo_a[i] = fiteo[0]
		fiteo_b[i] = fiteo[1]
		error_a[i] = error[0]
		error_b[i] = error[1]
		
		#Hago los graficos
		plt.figure(figsize=(18,9))
		plt.plot(espectro['pi_1'], espectro['pi_u'],
				'bo', label =' Datos',
				ms = 5, marker = '.')
		plt.plot(x_fit,	ajuste,
				'r-', lw = 3,
				label = ' a = '+ str('{:.2e}'.format(fiteo[0])) +
						' std_a = ' + str('{:.2e}'.format(error[0])) + '\n' + 
						' b = ' + str('{:.2e}'.format(fiteo[1])) +
						' std_b = ' + str('{:.2e}'.format(error[1])) + '\n' + 
						' R^2 = ' + str(round(r_cuadrado[i],2))
				)
		plt.yscale('log')
		plt.xscale('log')
		plt.axis([0.001, 10, 0.000001, 10])
		plt.title(nombre + " 1/L = " + str('{:.2e}'.format(obukhov[i])) + " Uo = " + str('{:.2e}'.format(u_star[i])))
		plt.xlabel("f*z/U")
		plt.ylabel("f*Su/(Uo^2*fi-e)")				#Para calcular los otros espectros cambiar fi-e
		plt.legend()
		plt.savefig(guardado + nombre + '.png')
		plt.close()
		i = i + 1
	#Guardo los datos en una tabla
	tabla = {
		'Fecha' : fecha,
		'1/L' : obukhov,
		'U' : u_medio,
		'Uo' : u_star,								#Para calcular los otros espectros tambien cambiar aca
		'a' : fiteo_a,
		'std_a' : error_a,
		'b' : fiteo_b,
		'std_b' : error_b,
		'R^2' : r_cuadrado
		}
	tabla_final = pd.DataFrame(data = tabla,)
	#Guardo la tabla
	tabla_final.to_csv(path_or_buf = guardado + 'Estadisticos.csv', index = False)

print('Todo salio bien')
input()