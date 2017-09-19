# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np
# carregando classificador Random Forest da biblioteca scikit-learn
from sklearn.ensemble import RandomForestClassifier
# carregando classificador SVm da biblioteca scikit-learn
from sklearn import svm
# carregando para gerar gráficos
import matplotlib.pyplot as plt
import matplotlib as mpl
# carregando para gerar matriz de confusão
from sklearn.metrics import confusion_matrix

# Load libraries
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

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
d_16=pd.read_csv("../SoccerPrediction/Data/15.16.csv")

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

df_16 = pd.DataFrame(d_16, columns = ['attendance','blocks_away_team','blocks_home_team','clearances_away_team',
'clearances_home_team','corners_away_team','corners_home_team','crosses_away_team','crosses_home_team','fouls_away_team','fouls_home_team',
'free_kicks_away_team','free_kicks_home_team','handballs_away_team','handballs_home_team','offsides_away_team','offsides_home_team',
'penalties_away_team','penalties_home_team','red_cards_away_team','red_cards_home_team','saves_away_team','saves_home_team',
'shots_off_target_away_team','shots_off_target_home_team','shots_on_target_away_team','shots_on_target_home_team','throw_ins_away_team','throw_ins_home_team',
'total_shots_away_team','total_shots_home_team','yellow_cards_away_team','yellow_cards_home_team','result','FTAG','FTHG','HomeTeam','AwayTeam'])

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

df_16['FTR'] = np.where((df_16['FTHG'] < df_16['FTAG']), 2, 1)
df_16['FTR'] = np.where((df_16['FTHG'] == df_16['FTAG']), 0, df_16['FTR'])

# invertendo os dataframes que devem ser invertidos
df_9 = df_9.iloc[::-1]
df_10 = df_10.iloc[::-1]
df_11 = df_11.iloc[::-1]
df_12 = df_12.iloc[::-1]
df_13 = df_13.iloc[::-1]
df_14 = df_14.iloc[::-1]
df_15 = df_15.iloc[::-1]
df_16 = df_16.iloc[::-1]

# criando um dataframe com todos os times que jogam as temporadas 05/06 e 06/07
df_unique = pd.DataFrame(df_6['HomeTeam'].unique())
# inicializando todos os times com 1500 de ELO
df_unique['ELO'] = 1500
# nomeando as colunas
df_unique.columns = ['Team', 'ELO']

###########################


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

# função que itera por todas as temporadas e calcula o ELO de cada time em cada momento
def calculate_elo(df):

    for index, row in df.iterrows():

        # inicializando as variáveis locais
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
        df.loc[index, 'OAE'] = old_away_elo
        df.loc[index, 'OHE'] = old_home_elo
        df.loc[index, 'ED'] = old_home_elo - old_away_elo


        # calculando o novo elo
        new_home_elo,new_away_elo = get_new_scores(old_home_elo,old_away_elo,h,a,f)
        # salvando no dataframe em uma nova coluna
        #df.loc[index, 'new away ELO'] = new_away_elo
        #df.loc[index, 'new home ELO'] = new_home_elo

        # guardando o novo elo dos times no dataframe com times e elos
        df_unique.loc[df_unique.Team == home_team, 'ELO'] = new_home_elo
        df_unique.loc[df_unique.Team == away_team, 'ELO'] = new_away_elo

# função que atualiza o dataframe que guarda os times que estão disputando
# a temporada analisada
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

# função que calcula a forma atual de um time baseando-se nos resultados
# das últimas quatro partidas
def calculate_form(df):

    # definindo um dataframe local contendo todos os times desta temporada
    df_unique_local =  pd.DataFrame(df['HomeTeam'].unique())
    df_unique_local.columns = ['Team']

    # inicializando variáveis locais
    df['AF'] = 0      # awayteam form
    df['HF'] = 0      # hometeam form
    df['ARNM4'] = 0   # awayteam result n minus 4
    df['ARNM3'] = 0   # awayteam result n minus 3
    df['ARNM2'] = 0   # awayteam result n minus 2
    df['ARNM1'] = 0   # awayteam result n minus 1
    df['AGNM4'] = 0   # awayteam goals n minus 4
    df['AGNM3'] = 0   # awayteam goals n minus 3
    df['AGNM2'] = 0   # awayteam goals n minus 2
    df['AGNM1'] = 0   # awayteam goals n minus 1
    df['AGSNM4'] = 0  # awayteam goals suffered n minus 4
    df['AGSNM3'] = 0  # awayteam goals suffered n minus 3
    df['AGSNM2'] = 0  # awayteam goals suffered n minus 2
    df['AGSNM1'] = 0  # awayteam goals suffered n minus 1
    df['HRNM4'] = 0   # hometeam result n minus 4
    df['HRNM3'] = 0   # hometeam result n minus 3
    df['HRNM2'] = 0   # hometeam result n minus 2
    df['HRNM1'] = 0   # hometeam result n minus 1
    df['HGNM4'] = 0   # hometeam goals n minus 4
    df['HGNM3'] = 0   # hometeam goals n minus 3
    df['HGNM2'] = 0   # hometeam goals n minus 2
    df['HGNM1'] = 0   # hometeam goals n minus 1
    df['HGSNM4'] = 0  # hometeam goals suffered n minus 4
    df['HGSNM3'] = 0  # hometeam goals suffered n minus 3
    df['HGSNM2'] = 0  # hometeam goals suffered n minus 2
    df['HGSNM1'] = 0  # hometeam goals suffered n minus 1


    df['Used'] = 0
    temp_index = 0
    df_temp = df.copy()

    # iterando por todos os times da temporada
    for i, r in df_unique_local.iterrows():

        team = df_unique_local.loc[i,'Team']

        # todas as partidas que envolvem o time dessa iteração
        df_temp = df.loc[(df.HomeTeam == team) | (df.AwayTeam == team)]
        df_temp = df_temp.reset_index()

        # loop onde será calculada a forma e armazenada em um dataframe temp
        for index, row in df_temp.iterrows():

            hometeam = df_temp.loc[index, 'HomeTeam']
            awayteam = df_temp.loc[index, 'AwayTeam']

            # as 3 primeiras partidas são ignoradas
            if index > 3:

                # levantando os dados das últimas quatro partidas
                result_n_minus_4 = df_temp.loc[(index - 4),'FTR'].item()
                result_n_minus_3 = df_temp.loc[(index - 3),'FTR'].item()
                result_n_minus_2 = df_temp.loc[(index - 2),'FTR'].item()
                result_n_minus_1 = df_temp.loc[(index - 1),'FTR'].item()
                n_minus_4_hometeam = df_temp.loc[(index - 4),'HomeTeam']
                n_minus_3_hometeam = df_temp.loc[(index - 3),'HomeTeam']
                n_minus_2_hometeam = df_temp.loc[(index - 2),'HomeTeam']
                n_minus_1_hometeam = df_temp.loc[(index - 1),'HomeTeam']
                n_minus_4_awayteam = df_temp.loc[(index - 4),'AwayTeam']
                n_minus_3_awayteam = df_temp.loc[(index - 3),'AwayTeam']
                n_minus_2_awayteam = df_temp.loc[(index - 2),'AwayTeam']
                n_minus_1_awayteam = df_temp.loc[(index - 1),'AwayTeam']

                coeff_n_minus_4 = 0
                coeff_n_minus_3 = 0
                coeff_n_minus_2 = 0
                coeff_n_minus_1 = 0

                # partida n-4
                if result_n_minus_4 == 0:

                    coeff_n_minus_4 = 1

                elif (result_n_minus_4 == 1 and n_minus_4_hometeam == team) or (result_n_minus_4 == 2 and n_minus_4_awayteam == team):

                    coeff_n_minus_4 = 3

                else:

                    coeff_n_minus_4 = 0

                # partida n-3
                if result_n_minus_3 == 0:

                    coeff_n_minus_3 = 1

                elif (result_n_minus_3 == 1 and n_minus_3_hometeam == team) or (result_n_minus_3 == 2 and n_minus_3_awayteam == team):

                    coeff_n_minus_3 = 3

                else:

                    coeff_n_minus_3 = 0

                # partida n-2
                if result_n_minus_2 == 0:

                    coeff_n_minus_2 = 1

                elif (result_n_minus_2 == 1 and n_minus_2_hometeam == team) or (result_n_minus_2 == 2 and n_minus_2_awayteam == team):

                    coeff_n_minus_2 = 3

                else:

                    coeff_n_minus_2 = 0

                # partida n-1
                if result_n_minus_1 == 0:

                    coeff_n_minus_1 = 1

                elif (result_n_minus_1 == 1 and n_minus_1_hometeam == team) or (result_n_minus_1 == 2 and n_minus_1_awayteam == team):

                    coeff_n_minus_1 = 3

                else:

                    coeff_n_minus_1 = 0


                form = 1*coeff_n_minus_4 + 1*coeff_n_minus_3 + 1*coeff_n_minus_2 + 1*coeff_n_minus_1

                temp_index = df.index[(df.HomeTeam == hometeam) & (df.AwayTeam == awayteam)]

                hometeam_temp = df.loc[temp_index, 'HomeTeam'].item()
                awayteam_temp = df.loc[temp_index, 'AwayTeam'].item()

                if hometeam_temp == team:

                    df.loc[temp_index, 'HF'] = form
                    df.loc[temp_index, 'HRNM4'] = coeff_n_minus_4
                    df.loc[temp_index, 'HRNM3'] = coeff_n_minus_3
                    df.loc[temp_index, 'HRNM2'] = coeff_n_minus_2
                    df.loc[temp_index, 'HRNM1'] = coeff_n_minus_1
                    df.loc[temp_index, 'Used'] = 1

                elif awayteam_temp == team:

                    df.loc[temp_index, 'AF'] = form
                    df.loc[temp_index, 'ARNM4'] = coeff_n_minus_4
                    df.loc[temp_index, 'ARNM3'] = coeff_n_minus_3
                    df.loc[temp_index, 'ARNM2'] = coeff_n_minus_2
                    df.loc[temp_index, 'ARNM1'] = coeff_n_minus_1
                    df.loc[temp_index, 'Used'] = 1

def calculate_past_games_features(df):

    # definindo um dataframe local contendo todos os times desta temporada
    df_unique_local =  pd.DataFrame(df['HomeTeam'].unique())
    df_unique_local.columns = ['Team']

    # inicializando features
    df['HHGR'] = 0  # hometeam_home_goals_ratio
    df['HSHGR'] = 0 # hometeam_suffered_home_goals_ratio
    df['AAGR'] = 0  # awayteam_away_goals_ratio
    df['ASAGR'] = 0 # awayteam_suffered_away_goals_ratio
    df['VHHR'] = 0  # victories hometeam home ratio
    df['DHHR'] = 0  # draws hometeam home ratio
    df['LHHR'] = 0  # losses hometeam home ratio
    df['VAAR'] = 0  # victories awayteam away ratio
    df['DAAR'] = 0  # draws awayteam away ratiio
    df['LAAR'] = 0  # losses awayteam away ratio

    # iterando por todos os times da temporada
    for i, r in df_unique_local.iterrows():

        # inicializando variáveis
        team = df_unique_local.loc[i,'Team']
        counter_home_matches = 1
        counter_away_matches = 1
        last_home_goal = 0
        last_away_goal = 0
        last_suff_home_goal = 0
        last_suff_away_goal = 0
        home_goal = 0
        away_goal = 0
        suff_home_goal = 0
        suff_away_goal = 0
        suff_home_goal_ratio = 0
        suff_away_goal_ratio = 0
        home_goal_ratio = 0
        away_goal_ratio = 0
        victories_hometeam_home = 0
        losses_hometeam_home = 0
        draws_hometeam_home = 0
        victories_hometeam_home_ratio = 0
        losses_hometeam_home_ratio = 0
        draws_hometeam_home_ratio = 0
        last_victories_hometeam_home = 0
        last_losses_hometeam_home = 0
        last_draws_hometeam_home = 0
        victories_awayteam_away = 0
        losses_awayteam_away = 0
        draws_awayteam_away = 0
        victories_awayteam_away_ratio = 0
        losses_awayteam_away_ratio = 0
        draws_awayteam_away_ratio = 0
        last_victories_awayteam_away = 0
        last_losses_awayteam_away = 0
        last_draws_awayteam_away = 0
        match_result = 0


        # todas as partidas que envolvem o time dessa iteração
        df_temp = df.loc[(df.HomeTeam == team) | (df.AwayTeam == team)]
        df_temp = df_temp.reset_index()

        # loop onde será calculada a forma e armazenada em um dataframe temp
        for index, row in df_temp.iterrows():

            hometeam = df_temp.loc[index, 'HomeTeam']
            awayteam = df_temp.loc[index, 'AwayTeam']

            if hometeam == team:

                if index == 0:

                    home_goal = df_temp.loc[index, 'FTHG'].item()
                    suff_home_goal = df_temp.loc[index, 'FTAG'].item()
                    home_goal_ratio = home_goal/counter_home_matches
                    suff_home_goal_ratio = suff_home_goal/counter_home_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        victories_hometeam_home = 1
                        victories_hometeam_home_ratio = 1

                    elif match_result == 2:

                        losses_hometeam_home = 1
                        losses_hometeam_home_ratio = 1

                    elif match_result == 0:

                        draws_hometeam_home = 1
                        draws_hometeam_home_ratio = 1

                elif index > 0:

                    home_goal = last_home_goal + df_temp.loc[index, 'FTHG'].item()
                    suff_home_goal = last_suff_home_goal + df_temp.loc[index, 'FTAG'].item()
                    home_goal_ratio = home_goal/counter_home_matches
                    suff_home_goal_ratio = suff_home_goal/counter_home_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        victories_hometeam_home = last_victories_hometeam_home + 1
                        victories_hometeam_home_ratio = victories_hometeam_home/counter_home_matches

                    elif match_result == 2:

                        losses_hometeam_home = last_losses_hometeam_home + 1
                        losses_hometeam_home_ratio = losses_hometeam_home/counter_home_matches

                    elif match_result == 0:

                        draws_hometeam_home = last_draws_hometeam_home + 1
                        draws_hometeam_home_ratio = draws_hometeam_home/counter_home_matches

                # atualizando variáveis para próxima iteração
                counter_home_matches = counter_home_matches + 1
                last_home_goal = home_goal
                last_suff_home_goal = suff_home_goal
                last_victories_hometeam_home = victories_hometeam_home
                last_losses_hometeam_home = losses_hometeam_home
                last_draws_hometeam_home = draws_hometeam_home

                # ultima partida de cada time nao precisa ser atualizada
                if index < 37:

                    # descobrindo qual o proximo jogo
                    # a informacao de razão entre gols semppre é guardada
                    # na linha referente à próxima partida
                    hometeam_next_game = df_temp.loc[index + 1, 'HomeTeam']
                    awayteam_next_game = df_temp.loc[index + 1, 'AwayTeam']

                    if hometeam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'HHGR'] = home_goal_ratio
                        df.loc[temp_index, 'HSHGR'] = suff_home_goal_ratio
                        df.loc[temp_index, 'VHHR'] = victories_hometeam_home_ratio
                        df.loc[temp_index, 'DHHR'] = draws_hometeam_home_ratio
                        df.loc[temp_index, 'LHHR'] = losses_hometeam_home_ratio

                    elif awayteam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'AAGR'] = away_goal_ratio
                        df.loc[temp_index, 'ASAGR'] = suff_away_goal_ratio
                        df.loc[temp_index, 'VAAR'] = victories_awayteam_away_ratio
                        df.loc[temp_index, 'DAAR'] = draws_awayteam_away_ratio
                        df.loc[temp_index, 'LAAR'] = losses_awayteam_away_ratio


            if awayteam == team:

                if index == 0:

                    away_goal = df_temp.loc[index, 'FTAG'].item()
                    suff_away_goal = df_temp.loc[index, 'FTHG'].item()
                    away_goal_ratio = away_goal/counter_away_matches
                    suff_away_goal_ratio = suff_away_goal/counter_away_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        losses_awayteam_away = 1
                        losses_awayteam_away_ratio = 1

                    elif match_result == 2:

                        victories_awayteam_away = 1
                        victories_awayteam_away_ratio = 1

                    elif match_result == 0:

                        draws_awayteam_away = 1
                        draws_awayteam_away_ratio = 1

                elif index > 0:

                    away_goal = last_away_goal + df_temp.loc[index, 'FTAG'].item()
                    suff_away_goal = last_suff_away_goal + df_temp.loc[index, 'FTHG'].item()
                    away_goal_ratio = away_goal/counter_away_matches
                    suff_away_goal_ratio = suff_away_goal/counter_away_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        losses_awayteam_away = last_losses_awayteam_away + 1
                        losses_awayteam_away_ratio = losses_awayteam_away/counter_away_matches

                    elif match_result == 2:

                        victories_awayteam_away = last_victories_awayteam_away + 1
                        victories_awayteam_away_ratio = victories_awayteam_away/counter_away_matches

                    elif match_result == 0:

                        draws_awayteam_away = last_draws_awayteam_away + 1
                        draws_awayteam_away_ratio = draws_awayteam_away/counter_away_matches

                counter_away_matches = counter_away_matches + 1
                last_away_goal = away_goal
                last_suff_away_goal = suff_away_goal
                last_victories_awayteam_away = victories_awayteam_away
                last_losses_awayteam_away = losses_awayteam_away
                last_draws_awayteam_away = draws_awayteam_away

                # ultima partida de cada time nao precisa ser atualizada
                if index < 37:

                    # descobrindo qual o proximo jogo
                    # a informacao de razão entre gols semppre é guardada
                    # na linha referente à próxima partida
                    hometeam_next_game = df_temp.loc[index + 1, 'HomeTeam']
                    awayteam_next_game = df_temp.loc[index + 1, 'AwayTeam']

                    if awayteam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'AAGR'] = home_goal_ratio
                        df.loc[temp_index, 'ASAGR'] = suff_home_goal_ratio
                        df.loc[temp_index, 'VAAR'] = victories_awayteam_away_ratio
                        df.loc[temp_index, 'DAAR'] = draws_awayteam_away_ratio
                        df.loc[temp_index, 'LAAR'] = losses_awayteam_away_ratio

                    elif hometeam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'HHGR'] = home_goal_ratio
                        df.loc[temp_index, 'HSHGR'] = suff_home_goal_ratio
                        df.loc[temp_index, 'VHHR'] = victories_hometeam_home_ratio
                        df.loc[temp_index, 'DHHR'] = draws_hometeam_home_ratio
                        df.loc[temp_index, 'LHHR'] = losses_hometeam_home_ratio

def plot_confusion_matrix(cm, classes = np.array(['Empate','Mandante','Visitante']),
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Resultado Real')
    plt.xlabel('Resultado Previsto')


def main():

    global df_15
    # calculando o ELO
    calculate_elo(df_6)
    # calculando a forma
    calculate_form(df_6)
    # calculando features
    calculate_past_games_features(df_6)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='West Brom',
    twentieth = 'Sunderland', first = 'Reading', second = 'Sheffield United',
    third = 'Watford')

    # calculando o ELO
    calculate_elo(df_7)
    # calculando a forma
    calculate_form(df_7)
    # calculando features
    calculate_past_games_features(df_7)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Sheffield United', nineteenth='Charlton',
    twentieth = 'Watford', first = 'Sunderland', second = 'Birmingham',
    third = 'Derby')

    # calculando o ELO
    calculate_elo(df_8)
    # calculando a forma
    calculate_form(df_8)
    # calculando features
    calculate_past_games_features(df_8)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Reading', nineteenth='Birmingham',
    twentieth = 'Derby', first = 'West Brom', second = 'Stoke',
    third = 'Hull')

    # calculando o ELO
    calculate_elo(df_9)
    # calculando a forma
    calculate_form(df_9)
    # calculando features
    calculate_past_games_features(df_9)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Newcastle', nineteenth='Middlesbrough',
    twentieth = 'West Brom', first = 'Wolves', second = 'Birmingham',
    third = 'Burnley')

    # calculando o ELO
    calculate_elo(df_10)
    # calculando a forma
    calculate_form(df_10)
    # calculando features
    calculate_past_games_features(df_10)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Burnley', nineteenth='Hull',
    twentieth = 'Portsmouth', first = 'Newcastle', second = 'West Brom',
    third = 'Blackpool')
    #print(df_unique)

    # calculando o ELO
    calculate_elo(df_11)
    # calculando a forma
    calculate_form(df_11)
    # calculando features
    calculate_past_games_features(df_11)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='Blackpool',
    twentieth = 'West Ham', first = 'QPR', second = 'Norwich',
    third = 'Swansea')

    # calculando o ELO
    calculate_elo(df_12)
    # calculando a forma
    calculate_form(df_12)
    # calculando features
    calculate_past_games_features(df_12)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Bolton', nineteenth='Blackburn',
    twentieth = 'Wolves', first = 'Reading', second = 'Southampton',
    third = 'West Ham')

    # calculando o ELO
    calculate_elo(df_13)
    # calculando a forma
    calculate_form(df_13)
    # calculando features
    calculate_past_games_features(df_13)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Wigan', nineteenth='Reading',
    twentieth = 'QPR', first = 'Cardiff', second = 'Hull',
    third = 'Crystal Palace')

    # calculando o ELO
    calculate_elo(df_14)
    # calculando a forma
    calculate_form(df_14)
    # calculando features
    calculate_past_games_features(df_14)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Norwich', nineteenth='Fulham',
    twentieth = 'Cardiff', first = 'Leicester', second = 'Burnley',
    third = 'QPR')

    # calculando o ELO
    calculate_elo(df_15)
    # calculando a forma
    calculate_form(df_15)
    # calculando features
    calculate_past_games_features(df_15)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Hull', nineteenth='Burnley',
    twentieth = 'QPR', first = 'Bournemouth', second = 'Watford',
    third = 'Norwich')

    # calculando o ELO
    #calculate_elo(df_16)
    # calculando a forma
    #calculate_form(df_16)
    # calculando features
    #calculate_past_games_features(df_16)
    # substituindo os times rebaixados pelos promovidos

    ### Análise das features ###

    # concatenando todas as temporadas em um só dataframe
    dfs = [df_9,df_10,df_11,df_12,df_13,df_14]
    df = pd.concat(dfs)
    print(df.groupby('FTR').size())

    df_new = df[['HHGR','HSHGR','AAGR','ASAGR','VHHR','DHHR','LHHR','VAAR','DAAR','LAAR','HF','AF','ED','Used','FTR']].copy()
    df_15_new = df_15[['HHGR','HSHGR','AAGR','ASAGR','VHHR','DHHR','LHHR','VAAR','DAAR','LAAR','HF','AF','ED','Used','FTR']].copy()

    train = df_new.loc[df_new.Used == 1]
    test = df_15_new.loc[df_15_new.Used == 1]

    # variável que guarda os valores dos resultados
    y = (train['FTR'])

    print('Quantidade de partidas utilizadas no treinamento:', len(train))
    print('Quantidade de partidas utilizadas no teste:',len(test))

    # criando conjunto de features (sem utilizar away_goals, home_goals e is_train)
    # 14 com OHE e OAE
    features = df_new.columns[0:13]
    print(features)



########### nem ideia

    clf1 = LogisticRegression(class_weight = 'balanced')
    clf2 = RandomForestClassifier(criterion='gini',n_estimators=1000, class_weight="balanced")
    #clf3 = ExtraTreesClassifier(n_estimators=1000, class_weight='balanced')
    clf4 = LinearDiscriminantAnalysis()
    clf5 = AdaBoostClassifier()

    eclf = VotingClassifier(estimators=[('LR', clf1), ('RF', clf2), ('LDA', clf4), ('ADB', clf5)], voting='hard')

    # Test options and evaluation metric
    seed = 10
    scoring = 'accuracy'


    # Algoritmos comparados
    models = []
    models.append(('ENS', eclf))
    models.append(('SGD', SGDClassifier(n_iter=1000, alpha=0.0001, class_weight = 'balanced')))
    #models.append(('LASSO', linear_model.Lasso())) #problema multiclass
    #models.append(('RIDGE', linear_model.Ridge())) #problema multiclass
    #models.append(('OvR', OneVsRestClassifier(eclf)))
    models.append(('LR', LogisticRegression(class_weight = 'balanced')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('BNB', BernoulliNB()))
    models.append(('SVM', SVC(class_weight='balanced')))
    models.append(('RF', RandomForestClassifier(criterion='gini',n_estimators=1000, class_weight="balanced")))
    models.append(('ERT', ExtraTreesClassifier(n_estimators=1000, class_weight='balanced')))
    models.append(('ADB', AdaBoostClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)))

    # loop que analisa cada algoritmo
    results = []
    names = []
    for name, model in models:
	    kfold = model_selection.KFold(n_splits=10, random_state=seed)
	    cv_results = model_selection.cross_val_score(model, train[features], y, cv=kfold, scoring=scoring)
	    results.append(cv_results)
	    names.append(name)
	    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	    print(msg)

	# Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Comparação dos Algoritmos')
    #plt.savefig(fig, bbox_inches='tight')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    #

    ## Prevendo resultados no dataset de teste - LR
    #print('LR')
    #lr = LogisticRegression(class_weight = 'balanced')
    #lr.fit(train[features], y)
    #predictions = lr.predict(test[features])
    #print(accuracy_score(test['FTR'], predictions))
    #print(classification_report(test['FTR'], predictions))
    #cm = pd.crosstab(test['FTR'], predictions, rownames=['Resultado real'], colnames=['Resultado simulado'])
    #print('Matriz de Confusão - LR')
    #print(cm)
    #print('-----------------------------------------------------')
    #
    #plt.matshow(cm)
    #plt.title('Matriz de Confusão - LR')
    #plt.colorbar()
    #plt.ylabel('Resultado Real')
    #plt.xlabel('Resultado Previsto')
    #plt.show()
    #
    ##Prevendo resultados no dataset de teste - RF
    print('RF')
    rf = RandomForestClassifier(n_estimators=1000,class_weight="balanced")
    rf.fit(train[features], y)
    predictions = rf.predict(test[features])
    print(accuracy_score(test['FTR'], predictions))
    print(classification_report(test['FTR'], predictions))
    cm = pd.crosstab(test['FTR'], predictions, rownames=['Resultado real'], colnames=['Resultado simulado'])
    print('Matriz de Confusão - RF')
    print(cm)
    importance_plot = rf.feature_importances_
    importance_plot = pd.DataFrame(importance_plot, index=train[features].columns,
                              columns=['Importance'])

    importance_plot['Std'] = np.std([tree.feature_importances_
                                for tree in rf.estimators_], axis=0)

    x = range(importance_plot.shape[0])
    y = importance_plot.ix[:, 0]
    yerr = importance_plot.ix[:, 1]

    plt.bar(x, y, yerr=yerr, align='center')
    plt.ylabel('Importância da variável')
    plt.xlabel('Variável')
    plt.title('Importância da variável')
    plt.show()
    #print('-----------------------------------------------------')
    #
    #plt.matshow(cm)
    #plt.title('Matriz de Confusão - RF')
    #plt.colorbar()
    #plt.ylabel('Resultado Real')
    #plt.xlabel('Resultado Previsto')
    #plt.show()
    #
    ## Prevendo resultados no dataset de teste - ERT
    #print('ERT')
    #ert = ExtraTreesClassifier(n_estimators=1000,class_weight = 'balanced')
    #ert.fit(train[features], y)
    #predictions = ert.predict(test[features])
    #print(accuracy_score(test['FTR'], predictions))
    #print(classification_report(test['FTR'], predictions))
    #cm = pd.crosstab(test['FTR'], predictions, rownames=['Resultado real'], colnames=['Resultado simulado'])
    #print('Matriz de Confusão - ERT')
    #print(cm)
    #print('-----------------------------------------------------')
    #
    #plt.matshow(cm)
    #plt.title('Matriz de Confusão - ERT')
    #plt.colorbar()
    #plt.ylabel('Resultado Real')
    #plt.xlabel('Resultado Previsto')
    #plt.show()
    #
    ## Prevendo resultados no dataset de teste - LDA
    #print('LDA')
    #lda.fit(train[features], y)
    #predictions = lda.predict(test[features])
    #print(accuracy_score(test['FTR'], predictions))
    #print(classification_report(test['FTR'], predictions))
    #cm = pd.crosstab(test['FTR'], predictions, rownames=['Resultado real'], colnames=['Resultado simulado'])
    #print('Matriz de Confusão - LDA')
    #print(cm)
    #print('-----------------------------------------------------')
    #
    #plt.matshow(cm)
    #plt.title('Matriz de Confusão - LDA')
    #plt.colorbar()
    #plt.ylabel('Resultado Real')
    #plt.xlabel('Resultado Previsto')
    #plt.show()
    #
    ## Prevendo resultados no dataset de teste - ADB
    #print('ADB')
    #adb = AdaBoostClassifier(n_estimators=1000)
    #adb.fit(train[features], y)
    #predictions = adb.predict(test[features])
    #print(accuracy_score(test['FTR'], predictions))
    #print(classification_report(test['FTR'], predictions))
    #cm = pd.crosstab(test['FTR'], predictions, rownames=['Resultado real'], colnames=['Resultado simulado'])
    #print('Matriz de Confusão - ADB')
    #print(cm)
    #print('-----------------------------------------------------')
    #
    #plt.matshow(cm)
    #plt.title('Matriz de Confusão - ADB')
    #plt.colorbar()
    #plt.ylabel('Resultado Real')
    #plt.xlabel('Resultado Previsto')
    #plt.show()



if __name__ == '__main__':
    main()
    #total = 0
    #for index, row in df_unique.iterrows():
    #    total += df_unique.loc[index,'ELO'].item()
    #print('total')
    #print(total)
    #df_6.to_csv('3.ELO.csv', sep=';')
