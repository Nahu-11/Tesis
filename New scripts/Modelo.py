#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


def resumen(datos):
    return print('Forma:\n', datos.shape,'\n\n',
                 'Columnas:\n', datos.columns,'\n\n',
                 'Tipos:\n', datos.dtypes,'\n\n',
                 'Datos nulos:\n', datos.isnull().sum(),'\n\n',
                 'Cabecera:\n', datos.head(10),'\n\n',
                 'Últimos datos:\n', datos.tail(10),'\n\n',
                 'Estadísticos:\n', datos.describe([0.05,0.25,0.5,0.75,0.95])
                )
def truncar(numero, decimales = 0):
    return np.trunc(numero*10**decimales)/(10**decimales)


# In[3]:


def lineal(x, a, b):
    linea = a*x + b
    return linea

def respiracion_1(t, ref, t0):
    # Poner t en Kelvin
    e0 = 135
    tref = 283.16
    resp = ref * np.exp(e0*(1/(tref-t0)-1/(t-t0)))
    return resp
def respiracion_2(t, ref, e0):
    # Poner t en Kelvin    
    tref = 283.16
    resp = ref * np.exp(e0*(1-tref/t)*(1/tref*8.31))
    return resp
def respiracion_3(t, ref, q10):
    # Poner t en Kelvin
    tref = 283.16
    resp = ref * q10**((t-tref)/10)
    return resp

def beta(b0, k, vpd, vpd0):
    if vpd > vpd0:
        beta = b0 * np.exp(-k*(vpd-vpd0))
    else:
        beta = b0
    return beta
def gpp(alfa, beta, rg):
    gpp = (alfa*beta*rg) / (alfa*rg + beta)
    return gpp
def nee(resp, gpp):
    nee = resp + gpp
    return nee


# In[4]:


def coef_determinacion(observaciones, predicciones):
    residuo = observaciones - predicciones
    ss_res = np.sum(residuo**2)
    ss_tot = np.sum((observaciones - np.mean(observaciones))**2)
    r_cuadrado = 1 - (ss_res/ss_tot)
    return r_cuadrado
def regresion(modelo, x, y):
    # No puede haber nungún NaN
    ajuste, covarianzas = curve_fit(modelo, x, y)
    predicciones = modelo(x, *ajuste)
    errores = np.sqrt(np.diag(covarianzas))
    r_cuadrado = coef_determinacion(y, predicciones)
    residuo = y - predicciones
    return predicciones, ajuste, errores, r_cuadrado, residuo


# In[5]:


def metadata(ejex, ejey):
    # plt.title(titulo, fontsize = 60, fontweight='bold')
    plt.xlabel(ejex, fontsize = 35)
    plt.ylabel(ejey, fontsize = 35)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    return
def cifras_signif(i, cifras):
    texto = str('{:g}'.format(float('{:.{p}g}'.format(i, p = cifras))))
    return texto
def grafico_modelo(x, y, predicciones, ajuste,
                   errores, r_cuad, nombres,
                   j = 3
                  ):
    etiqueta = ''
    iterador = list(range(len(nombres)))
    for i in iterador:
        valor = cifras_signif(ajuste[i], j)
        error = cifras_signif(errores[i], j)
        etiqueta = (etiqueta
                    + ' ' + nombres[i] + ' = ' + valor
                    + '; std ' + nombres[i] + ' = ' + error
                    + '\n')
    etiqueta = etiqueta + ' R^2 = ' + cifras_signif(r_cuad, j)
    plt.plot(x, y, 'bo', markersize = 2)
    plt.plot(x, predicciones, 'r-',
             label = etiqueta
             )
    plt.legend(fontsize = 20)
    return
def grafico_residuos(x, res):
    plt.plot(x, res, 'bo', markersize = 2)
    plt.axhline(0, color = 'black', linestyle = '--')
    return


# In[6]:


def regresion_y_grafico(modelo, x, y, xlabel, ylabel, nombres):
    predicciones, ajuste, errores, r_cuadrado, res = regresion(modelo, x, y)
    plt.subplot(221)
    grafico_modelo(x, y, predicciones, ajuste,
                   errores, r_cuadrado, nombres
                  )
    metadata(xlabel, ylabel)
    plt.subplot(222)
    grafico_residuos(x, res)
    metadata(xlabel, ylabel)
    return ajuste, r_cuadrado, errores
def identidad(metodo, ajuste, validacion_x, validacion_y, lab, nombres):
    iterador = list(validacion_x.index)
    pred = list(range(len(validacion_x)))
    for i in iterador:
        j = iterador.index(i)
        pred[j] = metodo(validacion_x[i], ajuste[0], ajuste[1])
    recta, ajuste, errores, r_cuadrado, res = regresion(lineal,
                                                        validacion_y,
                                                        pred
                                                       )
    plt.subplot(223)
    grafico_modelo(validacion_y, pred, recta,
                   ajuste, errores, r_cuadrado,
                   nombres
                  )
    metadata(lab + ' Obs', lab + ' Pred')
    extremos = [validacion_y.min(), validacion_y.max()]
    plt.plot(extremos, extremos, 'g--')
    return pred, r_cuadrado, ajuste, errores


# In[7]:


def mbe(x, y):
    n = x.count()
    diff = y - x
    mbe = diff.sum() * (1/n)
    return mbe
def mae(x, y):
    n = x.count()
    diff = abs(y - x)
    mae = diff.sum() * (1/n)
    return mae
def mse(x, y):
    n = x.count()
    diff = (y - x)**2
    mse = diff.sum() * (1/n)
    return mse
def rmse(x, y):
    rmse = (mse(x, y))**(1/2)
    return rmse
def indice_acuerdo(x, y):
    diff = (y - x)**2
    long = (np.abs(x) + np.abs(y))**2
    d = 1 - diff.sum()/long.sum()
    return d
def tabla_metricas(nombres = []):
    cols = ['MBE','MAE', 'MSE', 'RMSE',
            'R2_aj', 'R2_val', 'Acuerdo',
            'a val', 'std a val', 'b val',
            'std b val', 'n_aj', 'n_val'
           ]
    cols = cols + nombres
    iterador = list(range(len(nombres)))
    for i in iterador:
        nombres[i] = 'std ' + nombres[i]
    cols = cols + nombres
    num = len(cols)
    datos = np.zeros((1, num))
    tabla = pd.DataFrame(data = datos, columns = cols)
    return tabla
def metricas(x, y, tabla, j = 3):
    mbe_f = cifras_signif(mbe(x, y), j)
    mae_f = cifras_signif(mae(x, y), j)
    mse_f = cifras_signif(mse(x, y), j)
    rmse_f = cifras_signif(rmse(x, y), j)
    indice_f = cifras_signif(indice_acuerdo(x, y), j)
    tabla['MBE'][0] = mbe_f
    tabla['MAE'][0] = mae_f
    tabla['MSE'][0] = mse_f
    tabla['RMSE'][0] = rmse_f
    tabla['Acuerdo'][0] = indice_f
    return tabla
def grafico_metricas(tabla, lab):
    ax1 = plt.subplot(224)
    ax1.bar(tabla.columns[:4],
            tabla[tabla.columns[:4]].iloc[0],
            color = 'red'
            )
    metadata('', lab)
    ax2 = ax1.twinx()
    metr = list(tabla[tabla.columns[4:7]].iloc[0].values)
    ceros = [0, 0, 0, 0]
    lista = ceros + metr
    ax2.bar(tabla.columns[:7],
            lista,
            color = 'blue',
           )
    ax2.set_ylim(0, 1)
    metadata('', '')
    return


# In[8]:


def analisis(metodo, x, y, validacion_x, validacion_y,
             xlab, ylab, nombres, tabla
            ):
    grafico = plt.figure(figsize = (36, 18)).subplots(2, 2)
    ajuste, r_ajuste, std_ajuste = regresion_y_grafico(metodo, x, y,
                                                       xlab, ylab,
                                                       nombres
                                                       )
    pred, r_validac, recta, std_recta = identidad(metodo, ajuste,
                                                  validacion_x,
                                                  validacion_y,
                                                  ylab, ['a', 'b']
                                                  )
    tabla['R2_aj'][0] = r_ajuste
    tabla['R2_val'][0] = r_validac
    resultados = metricas(validacion_y, pred, tabla)
    grafico_metricas(resultados, ylab)
    resultados['n_aj'][0] = x.count() 
    resultados['n_val'][0] = validacion_x.count()
    iterador = list(range(len(nombres)))
    for i in iterador:
        resultados[nombres[i]][0] = ajuste[i]
        resultados['std ' + nombres[i]][0] = std_ajuste[i]
    resultados['a val'][0] = recta[0]
    resultados['std a val'][0] = std_recta[0]
    resultados['b val'][0] = recta[1]
    resultados['std b val'][0] = std_recta[1]
    return grafico, resultados


# In[10]:


datos = pd.read_csv(
    'C:\\Users\\nahue\\Desktop\\Tesis_2\\Datos\\Completos_buenos.txt',
    #'C:\\Users\\BIOMET\\Desktop\\Tesis_2\\Datos\\Completos_buenos.txt',
    delimiter = '\t',
    decimal = '.',
    na_values = -9999,
    skiprows = [1],
    encoding = 'ascii'
    )


# In[11]:


resumen(datos)


# In[12]:


dias = np.array([0, 16, 46, 76,
                107, 137, 168, 199,
                229, 260, 290, 303
                ])
lista = list(range(1, len(dias)))
promedio = list(range(1, len(dias)))
for i in lista:
    desde = dias[i-1] * 48
    hasta = dias[i] * 48
    promedio[i-1] = datos['Tair'][desde:hasta].mean()
print(promedio)


# In[13]:


datos['Tair'].mean()


# In[14]:


x = list(range(2, 13))
plt.figure(figsize = (18, 9))
plt.plot(x, promedio, color = 'black')
metadata('Mes', '°C')
plt.axhline(datos['Tair'].mean(), ls = '--')
plt.legend(['Temperatura mensual', 'Promedio'], fontsize= 20)


# In[15]:


datos_validos = datos[(datos['Rg'] < 5)
                      & (datos['Ustar'] > 0.14167)
                      & pd.notna(datos['Tair'])
                      & (datos['NEE'] >= 0)
                     ]
datos_validos.describe()


# In[16]:


muestra_tot = datos_validos.sample(frac = 4/5, random_state = 1).sort_values('Tair')
validacion_tot = datos_validos.drop(muestra_tot.index)


# In[62]:


modelo = [respiracion_3, 'Ref', 'Q10']
tair_mod_1_tot, para_analisis = analisis(modelo[0],
                                         muestra_tot['Tair'].add(273.15),
                                         muestra_tot['NEE'],
                                         validacion_tot['Tair'].add(273.15),
                                         validacion_tot['NEE'],
                                         'K', 'micromol/(m^2 s)',
                                         modelo[1:3], tabla_metricas(modelo[1:3])
                                         )
titulo = 'Desde 0 Hasta 360'
para_analisis['Desde'] = muestra_tot['DoY'].min()
para_analisis['Hasta'] = muestra_tot['DoY'].max()
plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\Q10\\Tair\\'
            + titulo +'.png'
           )


# In[57]:


para_analisis.head()


# In[58]:


dias = np.arange(60, 420, 60)
iterador = list(range(1, len(dias)))
for i in iterador:
    desde = dias[i-1]
    hasta = dias[i]
    muestra_vent = muestra_tot[(muestra_tot['DoY'] > desde)
                           & (muestra_tot['DoY'] < hasta)
                           ]
    validacion_vent = validacion_tot[(validacion_tot['DoY'] > desde)
                                     & (validacion_tot['DoY'] < hasta)
                                     ]
    cant_mu = muestra_vent['NEE'].notnull().sum()
    cant_va = validacion_vent['NEE'].notnull().sum()
    if cant_mu > 9 and cant_va > 2:
        titulo = 'Desde ' + str(desde) + ' Hasta ' + str(hasta)
        tair_mod_1_ven, error_vent = analisis(modelo[0],
                                              muestra_vent['Tair'].add(273.15),
                                              muestra_vent['NEE'],
                                              validacion_vent['Tair'].add(273.15),
                                              validacion_vent['NEE'],
                                              'K', 'micromol/(m^2 s)',
                                              modelo[1:3], tabla_metricas(modelo[1:3])
                                             )
        plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\Q10\\Tair\\'
                    + titulo +'.png'
                   )
        error_vent['Desde'] = muestra_vent['DoY'].min()
        error_vent['Hasta'] = muestra_vent['DoY'].max()
        para_analisis = para_analisis.append(error_vent, ignore_index = True)


# In[59]:


para_analisis.head()


# In[60]:


dias = np.arange(90, 355, 5)
iterador = list(range(3, len(dias)))
for i in iterador:
    desde = dias[i-3]
    hasta = dias[i]
    muestra_vent = muestra_tot[(muestra_tot['DoY'] > desde)
                           & (muestra_tot['DoY'] < hasta)
                           ]
    validacion_vent = validacion_tot[(validacion_tot['DoY'] > desde)
                                     & (validacion_tot['DoY'] < hasta)
                                     ]
    cant_mu = muestra_vent['NEE'].notnull().sum()
    cant_va = validacion_vent['NEE'].notnull().sum()
    if cant_mu > 9 and cant_va > 2:
        titulo = 'Desde ' + str(desde) + ' Hasta ' + str(hasta)
        tair_mod_1_ven, error_vent = analisis(modelo[0],
                                              muestra_vent['Tair'].add(273.15),
                                              muestra_vent['NEE'],
                                              validacion_vent['Tair'].add(273.15),
                                              validacion_vent['NEE'],
                                              'K', 'micromol/(m^2 s)',
                                              modelo[1:3], tabla_metricas(modelo[1:3])
                                             )
        plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\Q10\\Tair\\'
                    + titulo +'.png'
                   )
        error_vent['Desde'] = muestra_vent['DoY'].min()
        error_vent['Hasta'] = muestra_vent['DoY'].max()
        para_analisis = para_analisis.append(error_vent, ignore_index = True)


# In[61]:


para_analisis.to_csv('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\Q10\\Tair\\Resultados.csv',
                        sep = '\t',
                        na_rep = -9999,
                        index = False,
                        encoding = 'ascii'
                        )


# In[30]:


def todo(predictora, modelo):
    muestra_tot = datos_validos.sample(frac = 4/5, random_state = 1).sort_values(predictora)
    validacion_tot = datos_validos.drop(muestra_tot.index)
    tair_mod_1_tot, para_analisis = analisis(modelo[0],
                                             muestra_tot[predictora].add(273.15),
                                             muestra_tot['NEE'],
                                             validacion_tot[predictora].add(273.15),
                                             validacion_tot['NEE'],
                                             'K', 'micromol/(m^2 s)',
                                             modelo[1:3], tabla_metricas(modelo[1:3])
                                             )
    titulo = 'Desde 0 Hasta 360'
    para_analisis['Desde'] = muestra_tot['DoY'].min()
    para_analisis['Hasta'] = muestra_tot['DoY'].max()
    plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\'
                + modelo[2] + '\\'
                + predictora + '\\'
                + titulo +'.png'
               )
    dias = np.arange(60, 420, 60)
    iterador = list(range(1, len(dias)))
    for i in iterador:
        desde = dias[i-1]
        hasta = dias[i]
        muestra_vent = muestra_tot[(muestra_tot['DoY'] > desde)
                               & (muestra_tot['DoY'] < hasta)
                               ]
        validacion_vent = validacion_tot[(validacion_tot['DoY'] > desde)
                                         & (validacion_tot['DoY'] < hasta)
                                         ]
        cant_mu = muestra_vent['NEE'].notnull().sum()
        cant_va = validacion_vent['NEE'].notnull().sum()
        if cant_mu > 9 and cant_va > 2:
            titulo = 'Desde ' + str(desde) + ' Hasta ' + str(hasta)
            tair_mod_1_ven, error_vent = analisis(modelo[0],
                                                  muestra_vent[predictora].add(273.15),
                                                  muestra_vent['NEE'],
                                                  validacion_vent[predictora].add(273.15),
                                                  validacion_vent['NEE'],
                                                  'K', 'micromol/(m^2 s)',
                                                  modelo[1:3], tabla_metricas(modelo[1:3])
                                                 )
            plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\'
                        + modelo[2] + '\\'
                        + predictora + '\\'
                        + titulo +'.png'
                       )
            error_vent['Desde'] = muestra_vent['DoY'].min()
            error_vent['Hasta'] = muestra_vent['DoY'].max()
            para_analisis = para_analisis.append(error_vent, ignore_index = True)
    dias = np.arange(90, 355, 5)
    iterador = list(range(3, len(dias)))
    for i in iterador:
        desde = dias[i-3]
        hasta = dias[i]
        muestra_vent = muestra_tot[(muestra_tot['DoY'] > desde)
                               & (muestra_tot['DoY'] < hasta)
                               ]
        validacion_vent = validacion_tot[(validacion_tot['DoY'] > desde)
                                         & (validacion_tot['DoY'] < hasta)
                                         ]
        cant_mu = muestra_vent['NEE'].notnull().sum()
        cant_va = validacion_vent['NEE'].notnull().sum()
        if cant_mu > 10 and cant_va > 5:
            titulo = 'Desde ' + str(desde) + ' Hasta ' + str(hasta)
            tair_mod_1_ven, error_vent = analisis(modelo[0],
                                                  muestra_vent[predictora].add(273.15),
                                                  muestra_vent['NEE'],
                                                  validacion_vent[predictora].add(273.15),
                                                  validacion_vent['NEE'],
                                                  'K', 'micromol/(m^2 s)',
                                                  modelo[1:3], tabla_metricas(modelo[1:3])
                                                 )
            plt.savefig('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\'
                        + modelo[2] + '\\'
                        + predictora + '\\'
                        + titulo +'.png'
                       )
            error_vent['Desde'] = muestra_vent['DoY'].min()
            error_vent['Hasta'] = muestra_vent['DoY'].max()
            para_analisis = para_analisis.append(error_vent, ignore_index = True)
    para_analisis.to_csv('C:\\Users\\nahue\\Desktop\\Tesis_2\\Modelos\\'
                         + modelo[2] + '\\'
                         + predictora
                         + '\\Resultados.csv',
                            sep = '\t',
                            na_rep = -9999,
                            index = False,
                            encoding = 'ascii'
                            )
    print('Listo!')
    return


# In[21]:


modelo = [respiracion_3, 'Ref', 'Q10']
predictora = 'Tsoil'
todo(predictora, modelo)


# In[24]:


modelo = [respiracion_3, 'Ref', 'Q10']
predictora = ['TS1', 'TS2', 'TS3']
for i in [0, 1, 2]
    todo(predictora[i], modelo)


# In[31]:


todo(predictora[1], modelo)


# In[32]:


todo(predictora[2], modelo)


# In[ ]:




