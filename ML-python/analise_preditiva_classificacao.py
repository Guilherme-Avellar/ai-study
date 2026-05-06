# Imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# para mostrar a tabela inteira nos prints, config do pandas.
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Carregar o dataset
df = pd.read_csv("Analises_preditivas_classificacao/fertilizer_prediction.csv")

# Exibir as primeiras linhas do DataFrame
# print(df.head())

# Mostra informações gerais sobre a base
# print(df.info())

'''
Mesmo não tendo valores ausente, outros problemas podem aparecer, como outliers ou dados
duplicados.
Precisa verificar mais informações relevantes à limpeza
'''

# Verificar a presença de dados duplicados
duplicates = df.duplicated().sum()
# print("Número de dados duplicados:", duplicates)

# Verificar a presença de outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title("Boxplot para detectar outliers")
plt.xticks(rotation=45)
# plt.show()

'''
Há apenas 1 outlier, no Potassium. Mas vou ignorar e não retira-lo, pois está pouco
dicrepante em relação ao valor máximo
'''

# Exploração da distribuição dos labels
plt.figure(figsize=(10, 6))
sns.countplot(x='Fertilizer Name', data=df)
plt.title("Distribuição dos Tipos de Fertilizantes")
plt.xticks(rotation=45)
# plt.show()

'''
A distribuição dos labels mostram que existem 7 classes possíveis, a conclusão é
que é um problema de classsificação multiclasse. Há também uma certa distribuição,
então a acurácia não é a melhor métrica.
A próxima etapa é investigar a correlação entre as variáveis. Se duas colunas forem
altamente corelacionadas, é interessante remover alguma das duas, para reduzir a
dimensão e facilitar facilitar a convergência do algoritmo preditivo.
'''

# Correlação entre as features numéricas
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=numerics).corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
# plt.show()

'''
As variáveis Temperature e Humidity estão fortemente correlacionadas, e as variaveis
Phosphorous e Nitrogen estão, em certa medida, correlacionadas também.
No momento não vou me ajustar isso, vou trabalhar com todas as variáveis sendo features 
para o modelo preditivo.
Nesse cenário a base não possui problemas, mas se houvessem dados faltantes ou duplicados
é possivel ajustar com os próximos comandos.
'''

# Remover dados duplicados, se houver. Não há, então o código a seguir não altera nada
df = df.drop_duplicates()
# Tratar outliers, se necessário (substituir por média, mediana, etc.)
# Exemplo: substituir outliers da coluna 'Temperature' pela mediana
median_temperature = df['Temperature'].median()
df['Temperature'] = (df['Temperature'].apply(lambda x: median_temperature if x > df['Temperature'].quantile(0.975) or x < df['Temperature'].quantile(0.025) else x))
# print(df.shape)

'''
Agora vai começar a engenharia de features, serão realizados os seguintes passos:
1. Separa as features x do label y;
2. Criar uma codificação para os dados em string representando-os em via LabelEncoder
(para dados ordinais ou labels) ou OneHotEncoder (para dados nominais);
3. Remover as colunas originais antes dos tratamentos;
4. Separa os dados em treino (80%) e teste (20%);
5. Normalizar as features numéricas, sempre "aprendendo" a fórmula do treino e aplicando
no treino e no teste.
'''

# Separando features e labels
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Label Encoder para a variável alvo
le = LabelEncoder()
y = le.fit_transform(y)

# Lista de colunas categóricas e aplicação de One-Hot Encoding
categorical_cols = ['Soil Type', 'Crop Type']
ohe = OneHotEncoder(handle_unknown='ignore')
X_encoded = pd.DataFrame(ohe.fit_transform(X[categorical_cols]).toarray())

X_encoded = X_encoded.add_prefix('OHE_')

# Removendo colunas categóricas originais
X = X.drop(categorical_cols, axis=1)

# Concatenando as features codificadas com as numéricas
X = pd.concat([X, X_encoded], axis=1)

# Dividindo os dados em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização das features numéricas
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_teste_scaled = scaler.transform(X_test)

'''
Agora é possível aplicar os dados tratados em modelos preditivos.
Vou aplicar em Regressão Logística e em KNN usando K = 9
'''

# Regressão Logística
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_teste_scaled)
print("Acurácia Regressão Logística:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# KNN
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_teste_scaled)
print("Acurácia KNN:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

'''
A Regressão Logística performou melhor que o KNN, como demonstrado. Isso indica que esse
problema pode ser mais bem identificado de forma linear. 
Agora vai ser aplicado o modelo SVM, para comparar o resultado com modelos mais complexos.
'''

# SVM com kernel RBF -> mais complexo
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_teste_scaled)
print("Acurácia SVM (RBF):", accuracy_score(y_test, y_pred_svm_rbf))
print(classification_report(y_test, y_pred_svm_rbf))

# SVM com kernel polinomial -> intermediário
svm_poly = SVC(kernel='poly')
svm_poly.fit(X_train_scaled, y_train)
y_pred_svm_poly = svm_poly.predict(X_teste_scaled)
print("Acurácia SVM (Polinomial):", accuracy_score(y_test, y_pred_svm_poly))
print(classification_report(y_test, y_pred_svm_poly))

# SVM com kernel linear -> o mais simples
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_linear.predict(X_teste_scaled)
print("Acurácia SVM (Linear):", accuracy_score(y_test, y_pred_svm_linear))
print(classification_report(y_test, y_pred_svm_linear))


'''
Como esperado o SVM com kernel linear obteve maior acurácia. Agora vou aplicar modelos
baseados em árvores, para comparação com os outros modelos.
'''

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_teste_scaled)
print("Acurácia Decision Tree:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=25)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_teste_scaled)
print("Acurácia Random Forest:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

'''
Entre os modelos baseados em árvores, a árvore de decisão performou melhor. Nesse caso
a robustez e complexidade do algoritmo de Floresta não necessária.
Dentre todos os modelos, a Árvore de Decisão foi a melhor, sendo perfeita em acurácia.
Porém há alguns pontos de atenção:
1. Um modelo com acurácia 100% pode indicar índicio de contaminação de treino e teste,
mas não foi o caso nesse treino.
2. A base de dados usada é muito pequena, menos de 100 exemplos, então o conjunto de teste
não possui uma robustez e representação significativa para podermos afirmar que isso se
aplicará na realidade
3. Nem sempre é possível conseguir modelos com uma performace tão alta como nesse exemplo.
'''

