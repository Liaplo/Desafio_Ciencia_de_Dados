import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.svm import SVC

df = pd.read_csv("desafio_manutencao_preditiva_treino.csv")

failureType = ('No Failure','Power Failure','Tool Wear Failure','Overstrain Failure',
'Random Failures','Heat Dissipation Failure')

#análise gráfica
print("Porcentagem\n")
percentage = df['failure_type'].value_counts(normalize=True)*100
print(percentage)
print()

cols = ("type","air_temperature_k","process_temperature_k","rotational_speed_rpm",
"torque_nm","tool_wear_min")

for label in cols:
	for types in failureType:
		plt.hist(df[df["failure_type"]==types][label],label=types, alpha=0.7, density=True)
	plt.title(label)
	plt.ylabel("Probability")
	plt.xlabel(label)
	plt.legend()
	plt.show()
	
#convertendo para valores númericos a coluna type
dict_type = {"H":1,"L":2,"M":3}
df.type = df.type.map(dict_type)

#conjunto de dados para treino e teste
#75% dos dados serão para treino e 25% para teste
train,test = np.split(df.sample(frac=1), [int(0.75*len(df))])

def scale_dataset(dataframe, oversample=False):
	#duas primeiras colunas não agregam nos resultados
	X = dataframe[dataframe.columns[2:-1]].values
	#coluna com os resultados
	y = dataframe[dataframe.columns[-1]].values
	
	X = StandardScaler().fit_transform(X)
	
	#usadas para ajustar a distribuição de classes de um conjunto de dados
	if oversample:
		ros = RandomOverSampler()
		X, y = ros.fit_resample(X, y)
	
	data = np.hstack((X, np.reshape(y, (-1, 1))))
	
	return data, X, y


train, X_train, y_train = scale_dataset(train, oversample=True)
test, X_test, y_test = scale_dataset(test)

#------------algoritmo SVC-------------------------------------------
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
#accuracy = TP+TN/TP+TN+FN+FP
#precision = TP/TP+FP
#recall = TP/TP+FN
#f1 score = 2*(precision*recall/(precision+recall))
print(classification_report(y_test, y_pred))
#-------------------------------------------------------------

df_teste = pd.read_csv("desafio_manutencao_preditiva_teste.csv")
df_teste.type = df_teste.type.map(dict_type)

x_teste_pred = df_teste[df_teste.columns[2:]].values
x_teste_pred = StandardScaler().fit_transform(x_teste_pred)

y_teste_pred = svm_model.predict(x_teste_pred)

#gera a predição solicitada
pd.DataFrame(data={'rowNUmber':np.array(range(1,3334)),'predictedValues':y_teste_pred}).to_csv("predicted.csv",index=False)











