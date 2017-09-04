# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np

# leitura dos CSVs
d_6 = pd.read_csv("../SoccerPrediction/Data/05.06.csv")
d_7 = pd.read_csv("../SoccerPrediction/Data/06.07.csv")

# criando dataframes
df_6 = pd.DataFrame(d_6, columns = ['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])
df_7 = pd.DataFrame(d_7, columns = ['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])

df_6['ELO'] = 1500
df_6['counter'] = 0

dfs = [df_6,df_7]
# dataframe contendo todos os times
df = pd.concat(dfs)

# separando em um dataframe onde cada time aparece apenas uma vez
df_unique_clubs = pd.DataFrame(df['HomeTeam'].unique())
# iniciando todos os clubes da primeira temp. com 1500 de elo
df_unique_clubs['ELO'] = 1500
# iniciando o contador de partidas em 0
df_unique_clubs['counter'] = 0

# função que calcula o ELO
def get_new_scores(home_elo,away_elo,fthg,ftag,ftr):
    # fthg = full time hometeam goals
    # ftag = full time away goals
    # ftr = full time result
    # K padrão para premier league
    K = 30

    # diferença de elo entre os times do ponto de vista do time mandante
    rating_diff_h = home_elo - away_elo
    rating_diff_a = away_elo - home_elo

    # o resultado esperado é baseado no ponto de vista do time com maior elo
    if rating_diff_h > 0:
        home_expected = 1 / (1 + 10**(-(rating_diff_h+100)/400))
        away_expected = 1 / (1 + 10**(+(rating_diff_h+100)/400))

    else:
        away_expected = 1 / (1 + 10**(-(rating_diff_h+100)/400))
        home_expected = 1 / (1 + 10**(+(rating_diff_h+100)/400))

    # calculando o parâmetro G e W
    if ftr == 1:
        goals_diff = fthg - ftag
        W_h = 1
        W_a = 0
        if goals_diff == 1:
            G = 1
        if goals_diff == 2:
            G = 1.5
        else:
            G = (11+goals_diff)/8

    if ftr == 2:
        goals_diff = ftag - fthg
        W_h = 0
        W_a = 1
        if goals_diff == 1:
            G = 1
        if goals_diff == 2:
            G = 1.5
        else:
            G = (11+goals_diff)/8

    if ftr == 0:
        W_h = 0.5
        W_a = 0.5
        G = 1

    # calculando os novos elos
    home_new_elo = home_elo + K*G*(W_h - home_expected)
    away_new_elo = away_elo + K*G*(W_h - away_expected)

    return home_new_elo,away_new_elo

def main():

    for index, row in df_6.iterrows():
        if index < 10:
            df_6['HomeELO'], df_6['AwayELO'] = get_new_scores(1500,1500,df_6['FTHG'],df_6['FTAG'],df_6['FTR'])
            print('oi')
        #else



if __name__ == '__main__':
    main()
