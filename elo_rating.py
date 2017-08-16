# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np

# leitura dos CSVs

d_0=pd.read_csv("../SoccerPrediction/Data/00.01.csv")
d_1=pd.read_csv("../SoccerPrediction/Data/01.02.csv")
d_2=pd.read_csv("../SoccerPrediction/Data/02.03.csv")
d_3=pd.read_csv("../SoccerPrediction/Data/03.04.csv")
d_4=pd.read_csv("../SoccerPrediction/Data/04.05.csv")
d_5=pd.read_csv("../SoccerPrediction/Data/05.06.csv")
d_6=pd.read_csv("../SoccerPrediction/Data/06.07.csv")

df_0 = pd.DataFrame(d_0, columns = ['HomeTeam','AwayTeam','FTR'])
df_1 = pd.DataFrame(d_1, columns = ['HomeTeam','AwayTeam','FTR'])
df_2 = pd.DataFrame(d_2, columns = ['HomeTeam','AwayTeam','FTR'])
df_3 = pd.DataFrame(d_3, columns = ['HomeTeam','AwayTeam','FTR'])
df_4 = pd.DataFrame(d_4, columns = ['HomeTeam','AwayTeam','FTR'])
df_5 = pd.DataFrame(d_5, columns = ['HomeTeam','AwayTeam','FTR'])
df_6 = pd.DataFrame(d_6, columns = ['HomeTeam','AwayTeam','FTR'])
