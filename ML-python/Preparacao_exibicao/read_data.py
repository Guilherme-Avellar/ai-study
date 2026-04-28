import pandas as pd
import numpy as np

# para mostrar a tabela inteira nos prints, config do pandas.
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Fazendo um data frame com o arquivo csv.
df = pd.read_csv("exemplo_limpeza_dados.csv")
print(df)
print()

# Imprime o número de colunas e linhas da tabela (data frame)
# print(df.shape)

# Imprime 5 linhas aleatórias da tabela
# print(df.sample(5))

# imprime informações sobre as colunas, Non-Null Cout, que são os dados não faltantes, e depois o Dtype que é o tipo da
# coluna, "str" é considerado string, então precisa ser tratado para depois ser usado no modelo preditivo
# print(df.info())

# exibe estatística básica da tabela (quantidade de linhas,mediana, desvio padrão, etc)
# print(df.describe())

# para mostrar gráficamente os dados faltantes:
# import missingno as msno
# import matplotlib.pyplot as plt
# msno.matrix(df)
# plt.show()

# remover dados duplicados e atribuir o resultado a uma nova variável
# df_unicos = df.drop_duplicates()
# print(df_unicos)

# remover dados duplicados na veriavel original
df.drop_duplicates(inplace=True)
# print(df)

# em vez de remover as linhas com dados faltantes é possível preencher com dados pertinentes preenchendo as
# colunas "Age" e "Salary" com a média ou mediana de suas colunas
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
# print(df)

# Para os algoritmos de machine learning funcionar, todos os valores precisam ser expressos em forma númerica.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# Sempre precisa tratar os dados categóricos, separar os dados únicos
# print(df["Grade"].unique())
# Para eliminar os números das colunas Purchased e Grade, a estratégia é:
# cria colunas novas para representação numérica de Purchased e Grade.
# colocar os valores únicos em um array de strings em ordem crescente, e o valor atribuido para a nova coluna é o
# indice da posição que a string está
# dessa forma, na coluna Purchased_encoder por exemplo, os valore 0, 1, 2 vão substituir '1st', '3th', '2nd'.
le = LabelEncoder()
df["Purchased_encoder"] = le.fit_transform(df["Purchased"])
df["Grade_encoder"] = le.fit_transform(df["Grade"])
# Para a coluna "Coutry" não se aplica ordem crescente. O que é feito é uma coluna para cada instancia da coluna,
# nesse caso uma coluna para ca país que tem na tabela, e coloca true no país que pertence a essa linha e false
# nos demais. Lembrando que true ou false o computador enxerga 0 como False, e diferente de zero True, então está
# transformando a informação em números da mesma forma.
aux = pd.get_dummies(df["Country"])
final = pd.concat([df, aux], axis=1)
# agora se remove as colunas que já originais que foram transformadas em números de final
final.drop(columns=["Country","Grade", "Purchased"], inplace=True)
# print(final)
# guardar o resultado dos dados tratados que estão em final em um csv
final.to_csv("dados_tratados.csv")

# tratando a escala dos dados com padronização
sc = StandardScaler()
final_padronizado = sc.fit_transform(final)
# tratando a escala dos dados com normalização entre 10 e 20. (pode ser outra escala como 0 e 1)
mn = MinMaxScaler(feature_range=(10, 20))
final_normalizado = mn.fit_transform(final)
# salvando o resultado em novos data frames
df_final_padronizado = pd.DataFrame(final_padronizado, columns=final.columns)
df_final_normalizado = pd.DataFrame(final_normalizado, columns=final.columns)
# print(df_final_padronizado)
# print(df_final_normalizado)
# salvar a padronização e a normalização em arquivos csv
df_final_padronizado.to_csv("dados_padronizados.csv")
df_final_normalizado.to_csv("dados_normalizados.csv")

