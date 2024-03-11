# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:39:23 2024
Script de exemplo de aplicação de Random Forest para classificação do índice de qualidade do ar de Olivais em Portugal
@author: Eduardo Fernandes Henriques
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  
from matplotlib.dates import DateFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



df = pd.read_csv( "C:/script python/machine learning/exemplo floresta aleatoria/df_poluentes_olivais.csv" ,  sep=";")
#CO não incluido pois tem muito NA
df = df[["date","NO2","O3","PM10","PM25","SO2","indPM10"] ]
df['ind'] = "NA"


df.loc[ np.where (df.indPM10=="Muito bom" )[0]  ,["ind"]] =0
df.loc[ np.where (df.indPM10=="Bom" )[0]  ,["ind"]] = 1
df.loc[ np.where (df.indPM10=="Medio" )[0]  ,["ind"]] = 2
df.loc[ np.where (df.indPM10=="Fraco" )[0]  ,["ind"]] = 3
df.loc[ np.where (df.indPM10=="Mau" )[0]  ,["ind"]] = 4


df= df.dropna()
dfbk =df
X = df[["NO2","O3","PM10","PM25","SO2"] ].to_numpy()
y = df[ ["ind"] ].to_numpy()
y = np.ravel(y)


#Dividir vetores ou matrizes em subconjunto aleatorios
#70% para o dataset de treinamento e 30 % para o dataset de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70)

data = pd.DataFrame( { 'NO2': df.NO2, 'O3': df.O3, 'PM10': df.PM10, 'PM25': df.PM25, 'SO2': df.SO2,
                      'indice':df.ind } )
#Imprimir as primeiras linhas
print(data.head())
# Criar o classifcador RF
#n_estimators define o número de árvores
clf = RandomForestClassifier(n_estimators = 100   )  
  

#Treinar o modelo com base no conjunto de treino
y_train=y_train.astype('int')
clf.fit(X_train, y_train)




#A calcular os valores de predição conforme o conjunto de teste   
y_pred = clf.predict(X_test)  

#Obtem os parametros para estimator
clf.get_params()
pr =clf.predict_proba(X_test   )


#Obter a acuracia do modelo
y_test = y_test.astype('int')  
print ("Acuracia do modelo: " ,clf.score(X_test,y_test) )


# Calcular a acuracia do modelo. Outro método
acuracia = metrics.accuracy_score(y_test, y_pred)
print("Acuracia do modelo: ", acuracia)


y_pred_comp = clf.predict(X)
#A gerar um df com valores originais e calculado pelo modelo
df_res = pd.DataFrame({'date': df.date, 'y_ori': y, 'y_mod' :y_pred_comp })
df_res['date'] = pd.to_datetime(df_res['date'])

df_melt = pd.melt(df_res,id_vars="date")

#A gerar um df com as probabilidades por variável
num = np.arange(0, len( pr),1 )

if (pr.shape[1] ==5 ):
    df_prob = pd.DataFrame( {'i':num, 'Muito bom': pr[:,0], 'Bom': pr[:,1], 'Medio': pr[:,2], 'Fraco': pr[:,3], 'Mau': pr[:,4] } )        


if (pr.shape[1] ==4 ):
    df_prob = pd.DataFrame( {'i':num, 'Muito bom': pr[:,0], 'Bom': pr[:,1], 'Medio': pr[:,2], 'Fraco': pr[:,3] } )        
    
if (pr.shape[1] ==3 ):
    df_prob = pd.DataFrame( {'i':num, 'Muito bom': pr[:,0], 'Bom': pr[:,1], 'Medio': pr[:,2]  } )        


df_prob_m= pd.melt(df_prob, id_vars="i"   )


fig, ax = plt.subplots(1,2, figsize=(10, 6))
fig.suptitle('Aplicação de Random Forest em Olivais')


#Primeiro subplot
date_form = DateFormatter("%m-%Y")
sns.scatterplot(ax=ax[0],data=df_melt,x='date',y='value',hue='variable',size='variable', palette=['blue','red'],sizes=(5,20)  )
fig.legend(title='Variável',
    labels=['Índice  original','Índice predito'],
    loc='lower left', #'upper rigth','lower left','center', 'best'
            bbox_to_anchor=(0.8, -0.2),#Plota fora do gráfico
            frameon=False)#Remove a borda

my_xticks = ["Muito bom","Bom","Medio","Fraco","Ruim" ]
ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2) )
ax[0].xaxis.set(label_text='Data')
ax[0].yaxis.set(ticks=[0,1,2,3,4], ticklabels=my_xticks, label_text='Índice de qualidade do ar')
ax[0].set_xlabel( fontsize=10,xlabel="Data")
ax[0].set_ylabel( fontsize=15,ylabel='Índice de qualidade do ar')
ax[0].tick_params(axis='x', labelsize=8)
ax[0].tick_params(axis='y', labelsize=8)
ax[0].set_title('Acurácia do modelo : '+ str(np.round(acuracia,2) ))

leg = ax[0].get_legend()
leg.legend_handles[0].set_color('red')
leg.legend_handles[1].set_color('blue')

#Segundo subplot
sns.boxplot(ax=ax[1],data=df_prob_m ,x= 'variable',y='value'  ,  color='blue') 
ax[1].set_xlabel('Variável', fontsize=15)
ax[1].set_ylabel('Probabilidade', fontsize=15)


fname = "C:/script python/machine learning/exemplo floresta aleatoria/indice de qualidade do ar2.png"
plt.savefig(fname,dpi=300  )
plt.close()
plt.figure().clear()
plt.cla()
plt.clf()


