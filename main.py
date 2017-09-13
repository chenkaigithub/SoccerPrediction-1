# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np
# carregando classificador Random Forest da biblioteca scikit-learn
from sklearn.ensemble import RandomForestClassifier
# carregando para gerar gráficos
import matplotlib.pyplot as plt
# carregando para gerar matriz de confusão
from sklearn.metrics import confusion_matrix

# leitura dos CSVs
# temporadas para inicialização do ELO
d_6 = pd.read_csv("../SoccerPrediction/Data/05.06.csv", sep=';')
d_7 = pd.read_csv("../SoccerPrediction/Data/06.07.csv")
d_8 = pd.read_csv("../SoccerPrediction/Data/07.08.csv")
# temporadas de treinamento
d_9=pd.read_csv("../SoccerPrediction/Data/08.09.csv")
d_10=pd.read_csv("../SoccerPrediction/Data/09.10.csv")
d_11=pd.read_csv("../SoccerPrediction/Data/10.11.csv")
d_12=pd.read_csv("../SoccerPrediction/Data/11.12.csv")
d_13=pd.read_csv("../SoccerPrediction/Data/12.13.csv")
d_14=pd.read_csv("../SoccerPrediction/Data/13.14.csv")
#temporada de teste
d_15=pd.read_csv("../SoccerPrediction/Data/14.15.csv")
#d_16=pd.read_csv("../SoccerPrediction/Data/15.16.csv")

#### VARIÁVEIS GLOBAIS ####

# criando dataframes
# temporadas para inicialização do ELO
df_6 = pd.DataFrame(d_6, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_7 = pd.DataFrame(d_7, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_8 = pd.DataFrame(d_8, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
# temporadas de treinamento
df_9 = pd.DataFrame(d_9, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

df_10 = pd.DataFrame(d_10, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

df_11 = pd.DataFrame(d_11, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

df_12 = pd.DataFrame(d_12, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

df_13 = pd.DataFrame(d_13, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

df_14 = pd.DataFrame(d_14, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])
#temporada de teste
df_15 = pd.DataFrame(d_15, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

#df_16 = pd.DataFrame(d_16, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
#'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
#'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
#'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
#'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
#'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

#convertendo o resultado do tipo 0-0 para D
df_9['FTR'] = np.where((df_9['FTHG'] < df_9['FTAG']), 2, 1)
df_9['FTR'] = np.where((df_9['FTHG'] == df_9['FTAG']), 0, df_9['FTR'])

df_10['FTR'] = np.where((df_10['FTHG'] < df_10['FTAG']), 2, 1)
df_10['FTR'] = np.where((df_10['FTHG'] == df_10['FTAG']), 0, df_10['FTR'])

df_11['FTR'] = np.where((df_11['FTHG'] < df_11['FTAG']), 2, 1)
df_11['FTR'] = np.where((df_11['FTHG'] == df_11['FTAG']), 0, df_11['FTR'])

df_12['FTR'] = np.where((df_12['FTHG'] < df_12['FTAG']), 2, 1)
df_12['FTR'] = np.where((df_12['FTHG'] == df_12['FTAG']), 0, df_12['FTR'])

df_13['FTR'] = np.where((df_13['FTHG'] < df_13['FTAG']), 2, 1)
df_13['FTR'] = np.where((df_13['FTHG'] == df_13['FTAG']), 0, df_13['FTR'])

df_14['FTR'] = np.where((df_14['FTHG'] < df_14['FTAG']), 2, 1)
df_14['FTR'] = np.where((df_14['FTHG'] == df_14['FTAG']), 0, df_14['FTR'])

df_15['FTR'] = np.where((df_15['FTHG'] < df_15['FTAG']), 2, 1)
df_15['FTR'] = np.where((df_15['FTHG'] == df_15['FTAG']), 0, df_15['FTR'])

#df_16['FTR'] = np.where((df_16['FTHG'] < df_16['FTAG']), 2, 1)
#df_16['FTR'] = np.where((df_16['FTHG'] == df_16['FTAG']), 0, df_16['FTR'])

# invertendo os dataframes que devem ser invertidos
df_9 = df_9.iloc[::-1]
df_10 = df_10.iloc[::-1]
df_11 = df_11.iloc[::-1]
df_12 = df_12.iloc[::-1]
df_13 = df_13.iloc[::-1]
df_14 = df_14.iloc[::-1]
df_15 = df_15.iloc[::-1]
#df_16 = df_16.iloc[::-1]

# criando um dataframe com todos os times que jogam as temporadas 05/06 e 06/07
df_unique = pd.DataFrame(df_6['HomeTeam'].unique())
# inicializando todos os times com 1500 de ELO
df_unique['ELO'] = 1500
# nomeando as colunas
df_unique.columns = ['Team', 'ELO']

###########################

count = 0

# função que calcula o ELO
def get_new_scores(home_elo,away_elo,fthg,ftag,ftr):
    # fthg = full time hometeam goals
    # ftag = full time away goals
    # ftr = full time result
    # constante para premier league
    K = 30
    G = 0
    W_h = 0
    W_a = 0

    # diferença de elo entre os times do ponto de vista do time mandante
    rating_diff_h = home_elo - away_elo

    # o resultado esperado é baseado no ponto de vista do time com maior elo
    if rating_diff_h > 0:
        home_expected = 1 / (1 + 10**(-(rating_diff_h+100)/400))
        away_expected = 1 - home_expected

    else:
        home_expected = 1 / (1 + 10**(+(rating_diff_h+100)/400))
        away_expected = 1 - home_expected

    # calculando o parâmetro G e W
    if ftr.item() == 1:
        goals_diff = fthg.item() - ftag.item()
        W_h = 1
        W_a = 0
        if goals_diff == 1:
            G = 1
        if goals_diff == 2:
            G = 1.5
        else:
            G = (11+goals_diff)/8

    elif ftr.item() == 2:
        goals_diff = ftag.item() - fthg.item()
        W_h = 0
        W_a = 1
        if goals_diff == 1:
            G = 1
        if goals_diff == 2:
            G = 1.5
        else:
            G = (11+goals_diff)/8

    elif ftr.item() == 0:
        W_h = 0.5
        W_a = 0.5
        G = 1

    # calculando os novos elos
    home_new_elo = home_elo + K*G*(W_h - home_expected)
    away_new_elo = away_elo + K*G*(W_a - away_expected)

    return home_new_elo,away_new_elo

def calculate_elo(df):

    for index, row in df.iterrows():

        old_home_elo = None
        old_away_elo = None
        home_team = None
        away_team = None

        # variáveis de interesse para o cálculo do ELO
        # FTHG = full time hometeam goals
        # FTAG = full time away goals
        # FTR = full time result
        h = df.loc[index,'FTHG']
        a = df.loc[index,'FTAG']
        f = df.loc[index,'FTR']

        # identificando o mandante e o visitante da linha em questão
        home_team = df.loc[index,'HomeTeam']
        away_team = df.loc[index,'AwayTeam']

        # procurando o elo antigo dos times no dataframe com times e elos
        # salvando os elos antigos em variáveis
        old_home_elo = df_unique.loc[df_unique.Team == home_team, 'ELO'].item()
        old_away_elo = df_unique.loc[df_unique.Team == away_team, 'ELO'].item()
        # salvando no dataframe em uma nova coluna
        df.loc[index, 'old away ELO'] = old_away_elo
        df.loc[index, 'old home ELO'] = old_home_elo


        # calculando o novo elo
        new_home_elo,new_away_elo = get_new_scores(old_home_elo,old_away_elo,h,a,f)
        # salvando no dataframe em uma nova coluna
        #df.loc[index, 'new away ELO'] = new_away_elo
        #df.loc[index, 'new home ELO'] = new_home_elo

        # guardando o novo elo dos times no dataframe com times e elos
        df_unique.loc[df_unique.Team == home_team, 'ELO'] = new_home_elo
        df_unique.loc[df_unique.Team == away_team, 'ELO'] = new_away_elo

def update_teams(eigheenth, nineteenth, twentieth, first, second, third):

    # identificando os times rebaixados
    elo_eigheenth = df_unique.loc[df_unique.Team == eigheenth, 'ELO'].item()
    elo_nineteenth = df_unique.loc[df_unique.Team == nineteenth, 'ELO'].item()
    elo_twentieth = df_unique.loc[df_unique.Team == twentieth, 'ELO'].item()

    # somando seus ELOs
    sum_elo_relegated = elo_eigheenth + elo_nineteenth + elo_twentieth

    # 1500 * 3 = 4500
    diff = 4500 - sum_elo_relegated

    # todos os demais times sofrerão um ajuste para que os times promovidos
    # possam começar a próxima temporada com 1500 de ELO
    decrease_per_team = diff/17

    # corrigindo os ELOs
    for index, row in df_unique.iterrows():

        team = df_unique.loc[index,'Team']

        # times que continuam na Premier League
        if team != eigheenth and team != nineteenth and team != twentieth:

            df_unique.loc[index,'ELO'] = df_unique.loc[index,'ELO'] - diff/17

        # times rebaixados
        else:

            df_unique.loc[index,'ELO'] = 1500

    # times rebaixados
    # eigheenth = decimo oitavo colocado da Premier league
    # nineteenth = decimo nono colocado da Premier league
    # twentieth = vigésimo colocado da Premier league

    # times promovidos
    # first = primeiro colocado da Football league
    # second = segundo colocado da Football league
    # third = terceiro colcoado da Football league

    df_unique.loc[df_unique.Team == eigheenth, 'Team'] = first
    df_unique.loc[df_unique.Team == nineteenth, 'Team'] = second
    df_unique.loc[df_unique.Team == twentieth, 'Team'] = third


def main():

    global df_15
    # calculando o ELO
    calculate_elo(df_6)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='West Brom',
    twentieth = 'Sunderland', first = 'Reading', second = 'Sheffield United',
    third = 'Watford')

    # calculando o ELO
    calculate_elo(df_7)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Sheffield United', nineteenth='Charlton',
    twentieth = 'Watford', first = 'Sunderland', second = 'Birmingham',
    third = 'Derby')

    # calculando o ELO
    calculate_elo(df_8)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Reading', nineteenth='Birmingham',
    twentieth = 'Derby', first = 'West Brom', second = 'Stoke',
    third = 'Hull')

    # calculando o ELO
    calculate_elo(df_9)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Newcastle', nineteenth='Middlesbrough',
    twentieth = 'West Brom', first = 'Wolves', second = 'Birmingham',
    third = 'Burnley')

    # calculando o ELO
    calculate_elo(df_10)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Burnley', nineteenth='Hull',
    twentieth = 'Portsmouth', first = 'Newcastle', second = 'West Brom',
    third = 'Blackpool')
    #print(df_unique)

    # calculando o ELO
    calculate_elo(df_11)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='Blackpool',
    twentieth = 'West Ham', first = 'QPR', second = 'Norwich',
    third = 'Swansea')

    # calculando o ELO
    calculate_elo(df_12)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Bolton', nineteenth='Blackburn',
    twentieth = 'Wolves', first = 'Reading', second = 'Southampton',
    third = 'West Ham')

    # calculando o ELO
    calculate_elo(df_13)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Wigan', nineteenth='Reading',
    twentieth = 'QPR', first = 'Cardiff', second = 'Hull',
    third = 'Crystal Palace')

    # calculando o ELO
    calculate_elo(df_14)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Norwich', nineteenth='Fulham',
    twentieth = 'Cardiff', first = 'Leicester', second = 'Burnley',
    third = 'QPR')

    # calculando o ELO
    calculate_elo(df_15)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Hull', nineteenth='Burnley',
    twentieth = 'QPR', first = 'Bournemouth', second = 'Watford',
    third = 'Norwich')

    # calculando o ELO
    #calculate_elo(df_16)
    # substituindo os times rebaixados pelos promovidos
    #update_teams(eigheenth='Hull', nineteenth='Burnley',
    #twentieth = 'QPR', first = 'Bournemouth', second = 'Watford',
    #third = 'Norwich')


    ### Análise das features ###

    # concatenando todas as temporadas em um só dataframe
    dfs = [df_9,df_10,df_11,df_12,df_13,df_14]
    df = pd.concat(dfs)

    # excluindo colunas do resultado e da quantidade de gols de cada time
    df = df.drop('FTHG', 1)
    df = df.drop('FTAG', 1)
    df = df.drop('HomeTeam', 1)
    df = df.drop('AwayTeam', 1)
    df = df.drop('result', 1)
    df_15 = df_15.drop('FTHG', 1)
    df_15 = df_15.drop('FTAG', 1)
    df_15 = df_15.drop('HomeTeam', 1)
    df_15 = df_15.drop('AwayTeam', 1)
    df_15 = df_15.drop('result', 1)
    df['FTR new'] = df['FTR']
    df_15['FTR new'] = df_15['FTR']
    df = df.drop('FTR', 1)
    df_15 = df_15.drop('FTR', 1)

    #separando treinamento e teste
    df['is_train'] = True
    df_15['is train'] = False

    print('Numero de Colunas')
    print(len(df.columns[0:35]))

    # dois dataframes diferentes
    train = df
    test = df_15

    print('Quantidade de partidas utilizadas no treinamento:', len(train))
    print('Quantidade de partidas utilizadas no teste:',len(test))

    # criando conjunto de features (sem utilizar away_goals, home_goals e is_train)
    features = df.columns[0:35]

    # variável que guarda os valores dos resultados
    y = (train['FTR new'])

    # criando classificador
    clf = RandomForestClassifier(criterion='gini',n_estimators=1000)

    # treinando o classificador para se adequar aos dados de teste
    clf.fit(train[features], y)

    # utilizando o classificador para prever os resultados dos dados de treinamento
    prev = [clf.predict(test[features])]

    # matriz confusão
    cm = pd.crosstab(test['FTR new'], prev, rownames=['Resultado real'], colnames=['Resultado simulado'])
    print('matriz de confusao')
    print(cm)
    importances = list(zip(train[features], clf.feature_importances_))
    print('features:')
    print(importances)
    df_features = pd.DataFrame(importances)
    df_features.to_csv('2.out_VariableImportance.csv', sep=',')
    cm.to_csv('1.out_ConfusionMatrix.csv', sep=',')

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


if __name__ == '__main__':
    main()
    #total = 0
    #for index, row in df_unique.iterrows():
    #    total += df_unique.loc[index,'ELO'].item()
    #print('total')
    #print(total)
    #df_6.to_csv('3.ELO.csv', sep=';')
