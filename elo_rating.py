# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np

# leitura dos CSVs
d_6 = pd.read_csv("../SoccerPrediction/Data/05.06.csv", sep=';')
d_7 = pd.read_csv("../SoccerPrediction/Data/06.07.csv")

# criando dataframes
df_6 = pd.DataFrame(d_6, columns = ['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])
df_7 = pd.DataFrame(d_7, columns = ['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])

dfs = [df_6,df_7]
# dataframe contendo todos os times
df = pd.concat(dfs)

# criando um dataframe com todos os times que jogam as temporadas 05/06 e 06/07
df_unique = pd.DataFrame(df_6['HomeTeam'].unique())
# inicializando todos os times com 1500 de ELO
df_unique['ELO'] = 1500
# nomeando as colunas
df_unique.columns = ['Team', 'ELO']



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

    #print(home_expected)
    #print(away_expected)
    #print(rating_diff_h)
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

    else:
        print('nope')

    print()
    # calculando os novos elos
    home_new_elo = home_elo + K*G*(W_h - home_expected)
    away_new_elo = away_elo + K*G*(W_a - away_expected)
    #print(away_expected)

    return home_new_elo,away_new_elo

def main():

    # temporada 05-06
    for index, row in df_6.iterrows():

        # variáveis de interesse para o cálculo do ELO
        # FTHG = full time hometeam goals
        # FTAG = full time away goals
        # FTR = full time result
        h = df_6.loc[index,'FTHG']
        a = df_6.loc[index,'FTAG']
        f = df_6.loc[index,'FTR']

        # identificando o mandante e o visitante da linha em questão
        home_team = df_6.loc[index,'HomeTeam']
        away_team = df_6.loc[index,'AwayTeam']

        # procurando o elo antigo dos times no dataframe com times e elos
        # salvando os elos antigos em variáveis
        old_home_elo = df_unique.loc[df_unique.Team == home_team, 'ELO'].item()
        old_away_elo = df_unique.loc[df_unique.Team == away_team, 'ELO'].item()


        # calculando o novo elo
        new_home_elo,new_away_elo = get_new_scores(old_home_elo,old_away_elo,h,a,f)

        # guardando o novo elo dos times no dataframe com times e elos
        df_unique.loc[df_unique.Team == home_team, 'ELO'] = new_home_elo
        df_unique.loc[df_unique.Team == away_team, 'ELO'] = new_away_elo

    # times rebaixados da temporada 05-06
    # 18. Birmingham City
    # 19. West Bromwich Albion
    # 20. Sunderland

    #times promovidos para a PL 06-07
    # 1. Reading
    # 2. Sheffield United
    # 3. Watford

    print(df_unique)

    # Reading ganha a pontuação do Birmingham City
    # Sheffield United ganha a pontuação do West Bromwich Albion
    # Watford ganha a pontuação do Sunderland

    df_unique.loc[df_unique.Team == 'Birmingham', 'Team'] = 'Reading'
    df_unique.loc[df_unique.Team == 'West Brom', 'Team'] = 'Sheffield United'
    df_unique.loc[df_unique.Team == 'Sunderland', 'Team'] = 'Watford'

    # temporada 06-07
    for index, row in df_6.iterrows():

        # variáveis de interesse para o cálculo do ELO
        # FTHG = full time hometeam goals
        # FTAG = full time away goals
        # FTR = full time result
        h = df_7.loc[index,'FTHG']
        a = df_7.loc[index,'FTAG']
        f = df_7.loc[index,'FTR']

        # identificando o mandante e o visitante da linha em questão
        home_team = df_7.loc[index,'HomeTeam']
        away_team = df_7.loc[index,'AwayTeam']

        # procurando o elo antigo dos times no dataframe com times e elos
        # salvando os elos antigos em variáveis
        old_home_elo = df_unique.loc[df_unique.Team == home_team, 'ELO'].item()
        old_away_elo = df_unique.loc[df_unique.Team == away_team, 'ELO'].item()


        # calculando o novo elo
        new_home_elo,new_away_elo = get_new_scores(old_home_elo,old_away_elo,h,a,f)

        # guardando o novo elo dos times no dataframe com times e elos
        df_unique.loc[df_unique.Team == home_team, 'ELO'] = new_home_elo
        df_unique.loc[df_unique.Team == away_team, 'ELO'] = new_away_elo

    print(df_unique)

if __name__ == '__main__':
    main()
    total = 0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    #df_6.to_csv('3.ELO.csv', sep=';')
    #print(df_unique)
