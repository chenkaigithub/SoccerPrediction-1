# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np
# carregando classificador Random Forest da biblioteca scikit-learn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# leitura dos CSVs
# temporadas de treinamento
d_8=pd.read_csv("../SoccerPrediction/Data/08.09.csv")
d_9=pd.read_csv("../SoccerPrediction/Data/09.10.csv")
d_10=pd.read_csv("../SoccerPrediction/Data/10.11.csv")
d_11=pd.read_csv("../SoccerPrediction/Data/11.12.csv")
d_12=pd.read_csv("../SoccerPrediction/Data/12.13.csv")
d_13=pd.read_csv("../SoccerPrediction/Data/13.14.csv")
d_14=pd.read_csv("../SoccerPrediction/Data/14.15.csv")
#temporada de teste
d_15=pd.read_csv("../SoccerPrediction/Data/15.16.csv")


#transformando em dataframes
df_8 = pd.DataFrame(d_8, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_9 = pd.DataFrame(d_9, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_10 = pd.DataFrame(d_10, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_11 = pd.DataFrame(d_11, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_12 = pd.DataFrame(d_12, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_13 = pd.DataFrame(d_13, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_14 = pd.DataFrame(d_14, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

df_15 = pd.DataFrame(d_15, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','away_goals','home_goals'])

#convertendo o resultado do tipo 0-0 para D
df_8['result'] = np.where((df_8['home_goals'] < df_8['away_goals']), 'V', 'M')
df_8['result'] = np.where((df_8['home_goals'] == df_8['away_goals']), 'E', df_8['result'])

df_9['result'] = np.where((df_9['home_goals'] < df_9['away_goals']), 'V', 'M')
df_9['result'] = np.where((df_9['home_goals'] == df_9['away_goals']), 'E', df_9['result'])

df_10['result'] = np.where((df_10['home_goals'] < df_10['away_goals']), 'V', 'M')
df_10['result'] = np.where((df_10['home_goals'] == df_10['away_goals']), 'E', df_10['result'])

df_11['result'] = np.where((df_11['home_goals'] < df_11['away_goals']), 'V', 'M')
df_11['result'] = np.where((df_11['home_goals'] == df_11['away_goals']), 'E', df_11['result'])

df_12['result'] = np.where((df_12['home_goals'] < df_12['away_goals']), 'V', 'M')
df_12['result'] = np.where((df_12['home_goals'] == df_12['away_goals']), 'E', df_12['result'])

df_13['result'] = np.where((df_13['home_goals'] < df_13['away_goals']), 'V', 'M')
df_13['result'] = np.where((df_13['home_goals'] == df_13['away_goals']), 'E', df_13['result'])

df_14['result'] = np.where((df_14['home_goals'] < df_14['away_goals']), 'V', 'M')
df_14['result'] = np.where((df_14['home_goals'] == df_14['away_goals']), 'E', df_14['result'])

df_15['result'] = np.where((df_15['home_goals'] < df_15['away_goals']), 'V', 'M')
df_15['result'] = np.where((df_15['home_goals'] == df_15['away_goals']), 'E', df_15['result'])

dfs = [df_8,df_9,df_10,df_11,df_12,df_13,df_14]

# concatenando todas as temporadas em um só dataframe
df = pd.concat(dfs)

#separando treinamento e teste
df['is_train'] = True
df_15['is train'] = False

# dois dataframes diferentes
train = df
test = df_15

print('Quantidade de partidas utilizadas no treinamento:', len(train))
print('Quantidade de partidas utilizadas no teste:',len(test))

# criando conjunto de features (sem utilizar away_goals, home_goals e is_train)
features = df.columns[0:33]

# variável que guarda os valores dos resultados
y = (train['result'])

# criando classificador
clf = RandomForestClassifier(criterion='gini',n_estimators=1000)

# treinando o classificador para se adequar aos dados de teste
clf.fit(train[features], y)

# utilizando o classificador para prever os resultados dos dados de treinamento
prev = [clf.predict(test[features])]

# matriz confusão
cm = pd.crosstab(test['result'], prev, rownames=['Resultado real'], colnames=['Resultado simulado'])
print('matriz de confusao')
print(cm)
importances = list(zip(train[features], clf.feature_importances_))
print('features:')
print(importances)
df_features = pd.DataFrame(importances)
df_features.to_csv('2.out_VariableImportance.csv', sep=',')
cm.to_csv('1.out_ConfusionMatrix.csv', sep=',')
cm.to_csv('4.out_ConfusionMatrix.csv', sep=',')

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Matriz de Confusão')
plt.colorbar()
plt.ylabel('Resultado Real')
plt.xlabel('Resultado Previsto')
plt.show()

#importance for plot

importance_plot = clf.feature_importances_
importance_plot = pd.DataFrame(importance_plot, index=train[features].columns,
                          columns=['Importance'])

importance_plot['Std'] = np.std([tree.feature_importances_
                            for tree in clf.estimators_], axis=0)

x = range(importance_plot.shape[0])
y = importance_plot.ix[:, 0]
yerr = importance_plot.ix[:, 1]

plt.bar(x, y, yerr=yerr, align='center')
plt.ylabel('Importância da variável')
plt.xlabel('Variável')
plt.title('Importância da variável')
plt.show()
