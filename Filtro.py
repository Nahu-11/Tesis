#Importo librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Importo la base de datos
data = pd.read_csv(
        'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Datos.csv',
        delimiter = ',',
        decimal = '.',
        engine = 'python'
        )
data = data.drop(columns=['Unnamed: 61'])

#Agrego columna con el path de los espectros
total = data['T_begin        '].count()
archivo =  list(range(total))
for x in range(total):
    archivo[x] = ("Marchi_" + 
        data['T_begin        '][x][0:2] + "_" + #Dia
        data['T_begin        '][x][3:5] + "_" + #Mes
        data['T_begin        '][x][11:13] + #Hora inicial
        data['T_begin        '][x][14:16] + "-" + #Minutos inicial
        data['T_end          '][x][11:13] + #Hora final
        data['T_end          '][x][14:16] + ".spe") #Minutos final

data['Path'] = archivo

#El z del instrumento es 2.85m
data['1/L'] = data['z/L            ']/2.85

#Filtro los datos buenos
ustar = data[(data['Flag(ustar)    ']==1)]
hts = data[(data['Flag(HTs)      ']==1)]
lve = data[(data['Flag(LvE)      ']==1)]
wco2 = data[(data['Flag(wCO2)     ']==1)]

ustar_n = ustar[(abs(ustar['1/L'])<0.0001) & (ustar['ustar[m/s]     ']>0.31) & (ustar['ustar[m/s]     ']<0.55)]
hts_n = hts[(abs(hts['1/L'])<0.0001)]
lve_n = lve[(abs(lve['1/L'])<0.0001)]
wco2_n = wco2[(abs(wco2['1/L'])<0.0001)]

#Creo directorios para guardar los gráficos y tablas
direccion = 'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Neutral'
if not os.path.exists(direccion):
	os.makedirs(direccion)

#Guardo la tabla
ustar_n.to_csv(path_or_buf = direccion + '\\ustar.csv', index = False)
hts_n.to_csv(path_or_buf = direccion + '\\hts.csv', index = False)
lve_n.to_csv(path_or_buf = direccion + '\\lve.csv', index = False)
wco2_n.to_csv(path_or_buf = direccion + '\\wco2.csv', index = False)

ustar_i = ustar[(ustar['1/L']<-0.045) & (ustar['1/L']>-0.55) & (ustar['ustar[m/s]     ']>0.35) & (ustar['ustar[m/s]     ']<0.5)]
hts_i = hts[(hts['1/L']<-0.35) & (hts['1/L']>-0.6)]
lve_i = lve[(lve['1/L']<-0.35) & (lve['1/L']>-0.6)]
wco2_i = wco2[(wco2['1/L']<-0.35) & (wco2['1/L']>-0.6)]

#Creo directorios para guardar los gráficos y tablas
direccion = 'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Inestable'
if not os.path.exists(direccion):
	os.makedirs(direccion)

#Guardo la tabla
ustar_i.to_csv(path_or_buf = direccion + '\\ustar.csv', index = False)
hts_i.to_csv(path_or_buf = direccion + '\\hts.csv', index = False)
lve_i.to_csv(path_or_buf = direccion + '\\lve.csv', index = False)
wco2_i.to_csv(path_or_buf = direccion + '\\wco2.csv', index = False)

ustar_e = ustar[(ustar['1/L']>0.015) & (ustar['1/L']<0.6) & (ustar['ustar[m/s]     ']>0.305) & (ustar['ustar[m/s]     ']<0.55)]
hts_e = hts[(hts['1/L']>0.35) & (hts['1/L']<0.6)]
lve_e = lve[(lve['1/L']>0.35) & (lve['1/L']<0.6)]
wco2_e = wco2[(wco2['1/L']>0.35) & (wco2['1/L']<0.6)]

#Creo directorios para guardar los gráficos y tablas
direccion = 'C:\\Users\\nahue\\Desktop\\Tesis\\Datos\\Estable'
if not os.path.exists(direccion):
	os.makedirs(direccion)

#Guardo la tabla
ustar_e.to_csv(path_or_buf = direccion + '\\ustar.csv', index = False)
hts_e.to_csv(path_or_buf = direccion + '\\hts.csv', index = False)
lve_e.to_csv(path_or_buf = direccion + '\\lve.csv', index = False)
wco2_e.to_csv(path_or_buf = direccion + '\\wco2.csv', index = False)

print('Todo salio bien')
input()