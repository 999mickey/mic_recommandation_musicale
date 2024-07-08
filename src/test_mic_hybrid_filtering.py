from models.mic_filtering_class import mic_hybrid_filtering

import time
import os
import pandas as pd
from input_variables_setter import input_variables_setter
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
beep(1)

start = time.time()

input_path = '../Data/'

file_name = 'twitter_track_sentiments.csv'
file_name = 'user_and_track_sentiment.csv'
file_name = 'spotify_user_track_norm.csv'
#file_name = "simulation5-10.csv"
file_name = 'spotify_user_track_reduced10.01.csv'
file_name = "simulation.csv"
file_name = "simulationcurrent.csv"
#file_name = "simulationbig.csv"
#file_name = "merge.csv"

#file_name = "simulation5-10.csv"
file_name = "simulation_u3-t20_g2_m6.csv"
file_name = "simulation_u10-t20_g5_m10.csv"

#file_name = "ratings_flrd.csv"


my_mic_hybrid_filtering = mic_hybrid_filtering()
input_variables_setter(file_name,my_mic_hybrid_filtering)
my_mic_hybrid_filtering.set_data(input_path+file_name)
print(my_mic_hybrid_filtering.data_frame.head(len(my_mic_hybrid_filtering.data_frame)))

my_mic_hybrid_filtering.clean_columns()
userId = my_mic_hybrid_filtering.default_user
#userId = "user30_pop_classic_metal"
my_mic_hybrid_filtering.data_frame.fillna(0, inplace=True)
my_mic_hybrid_filtering.generate_dateset_autofold()
user_id = "usr2_metal"
user_id ="usr9_rap_pop_jazz_classic_metal"
trackid = "track3_jazz"



ret = my_mic_hybrid_filtering.get_and_remove_from_dataframe(user_id,trackid)
print("ret = ",ret)
print("ret = ",ret.iloc[0])



filter2 =  mic_hybrid_filtering()
filter2.data_frame = ret; 
filter2.item_key_name = my_mic_hybrid_filtering.item_key_name
filter2.user_key_name = my_mic_hybrid_filtering.user_key_name
filter2.target_key_name = my_mic_hybrid_filtering.target_key_name

filter2.generate_dateset_autofold()

from surprise import NormalPredictor
normpred = NormalPredictor()
pred = my_mic_hybrid_filtering.predict(user_id,normpred,5) 
print(pred)

normpred = NormalPredictor()
#pred = my_mic_hybrid_filtering.predictwith_testset(filter2.data.build_full_trainset(),normpred)
#pred = my_mic_hybrid_filtering.predictwith_testset(filter2.compute_antitraintest_with_threshold(user_id,0),normpred)
pred = my_mic_hybrid_filtering.predictwith_testset(filter2.compute_traintest(user_id),normpred)




print(pred)


#llist = []
#llist.append((user_id, train_set.to_raw_iid(track), moyenne))

print(my_mic_hybrid_filtering.data_frame.head(len(my_mic_hybrid_filtering.data_frame)))


#my_mic_hybrid_filtering.compute_trainset()
exit()


#userId= '464336760142aecf41fc6ab6535171d3'
#userId= 'user29_classic'
userId = '464336760142aecf41fc6ab6535171d3'

from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

#normpred = NormalPredictor()
#param_grid = {}
#grid_search = GridSearchCV(NormalPredictor(), param_grid, measures=['rmse','mae'], cv=3)



##################################### NormalPredictor
"""from surprise import NormalPredictor
predictor = NormalPredictor()
param_grid ={}

ret = my_mic_hybrid_filtering.predictor_ajustement(NormalPredictor(),param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)
exit()
###"""
#print("myfilter.best_params_predictor = \n",myfilter.best_params_predictor)
"""print("++++++++++++++Predictions NormalPredictor() pour l 'utilisateur [",userId1,"]")
predictor = NormalPredictor(**myfilter.best_params_predictor)

pred = myfilter.predict(userId1,predictor,5) 
print(pred)


pred = myfilter.predict_with_train_split(userId1,predictor,5) 
print(pred)

print("++++++++++++++Fin des predictions NormalPredictor() pour l 'utilisateur [",userId1,"]")

#"""


from surprise import NormalPredictor
pred = NormalPredictor()
from surprise.prediction_algorithms.knns import KNNBasic
pred =KNNBasic()

my_mic_hybrid_filtering.evaluate_predictor(pred)

predictions = my_mic_hybrid_filtering.predict(userId,pred,10)

maeres , rmse = my_mic_hybrid_filtering.evaluate_predictions(predictions)
print(predictions.shape)
print(predictions.head(10))
exit()  
"""
from surprise import SVD
from surprise import Reader
from surprise import Dataset
"""
reader = Reader(rating_scale=(0, 5))
df = my_mic_hybrid_filtering.data_frame
df_surprise = Dataset.load_from_df(df[["user_id", "title", "rating"]], reader=reader)
print(df_surprise)

pred = SVD()

my_mic_hybrid_filtering.evaluate_predictor(pred)

param_grid = {'n_factors': [100,150],
              'n_epochs': [20,25,30],
              'lr_all':[0.005,0.01,0.1],
              'reg_all':[0.02,0.05,0.1]}

param_grid = {'n_factors': [10,15],
                        'n_epochs': [20,25],
                        'lr_all':[0.005,0.1],
                        'reg_all':[0.02,0.1]}



ret = my_mic_hybrid_filtering.predictor_ajustement(SVD,param_grid)
print("paramètres optimaux pour SVD")
print(ret)

#nfactor = ret["n_factors"]
#nepochs = ret["n_epochs"]
#lrall = ret["lr_all"]
#regall = ret["reg_all"]

exit()
###"""
print("++++++++++++++Predictions SVD() pour l 'utilisateur [",userId,"]  ==>")
#predictor = SVD()
#predictor = SVD(**my_mic_hybrid_filtering.best_params_predictor)
predictor = SVD(n_factors= 100, n_epochs= 20, lr_all=0.005, reg_all= 0.1)


my_mic_hybrid_filtering.evaluate_predictor(pred)




predictions = my_mic_hybrid_filtering.predict(userId,pred,10)
print(predictions.shape)
print(predictions.head(10))



exit()


######################spotify_user_track_reduced10.01.csv  171.8 M
#myfilter.data_frame = myfilter.data_frame.sample(frac=0.01,random_state=666)
#"""   
###############################'user_and_track_sentiment.csv' 428.6 M
#on enlève les hashtag majoritaires
"""myfilter.data_frame = myfilter.data_frame[myfilter.data_frame['hashtag'] != 'nowplaying']
myfilter.data_frame.info()
print("hashtag les plus courants => ")
print(myfilter.data_frame["hashtag"].unique())
myfilter.clean_columns()
myfilter.data_frame.info()
global_hashtag = 'punk'
ret = myfilter.get_random_row_from('hashtag',global_hashtag)
##"""
############################'simulation5-10.csv'
##################################################
myfilter. display_tracks_and_users_num()
#choix 
# de l'utilisateur 
userId1 = myfilter.get_random_user_id()
#userId1 = 'user0'
userId1 = myfilter.default_user

outcsv_path = '../Data/CsvResults/'
number_of_line_todump = 10

print(myfilter.get_user_description(userId1))

myfilter.compute_antitraintest(userId1)

myfilter.compute_antitraintest1(userId1)
exit()


#######################################SVD
from surprise import SVD
predictor = SVD()
"""param_grid = {'n_factors': [100,150],
              'n_epochs': [20,25,30],
              'lr_all':[0.005,0.01,0.1],
              'reg_all':[0.02,0.05,0.1]}
"""
param_grid = {'n_factors': [10,15],
                        'n_epochs': [20,25],
                        'lr_all':[0.005,0.1],
                        'reg_all':[0.02,0.1]}

ret = myfilter.predictor_ajustement(predictor,param_grid)
print("paramètres optimaux pour Normal Pred")
print(ret)

nfactor = ret["n_factors"]
nepochs = ret["n_epochs"]
lrall = ret["lr_all"]
regall = ret["reg_all"]


print("++++++++++++++Predictions SVD() pour l 'utilisateur [",userId1,"]  ==>")
#predictor = SVD()
predictor = SVD(**myfilter.best_params_predictor)
pred = myfilter.predict(userId1,predictor,5) 
ret = pred
print(ret)

predictor = SVD(**myfilter.best_params_predictor)
pred = myfilter.predict_with_train_split(userId1,predictor,5) 
ret = pred
print(ret)

print("++++++++++++++Fin des predictions SVD() pour l 'utilisateur [",userId1,"]")
exit()
##############################KNNBasic
"""from surprise.prediction_algorithms.knns import KNNBasic
sim_options = {'name': 'cosine',
               'user_based': False
               }
predictor = KNNBasic(sim_options=sim_options)
pred = myfilter.predict(userId1,predictor) 
ret = pred
print(ret)
print("Fin des predictions KNN() pour l 'utilisateur [",userId1,"]")
"""
####################################BaselineOnly
####################################KNNBaseline
####################################Co-clustering
"""from sklearn.cluster import SpectralCoclustering 
predictor = SpectralCoclustering()
pred = myfilter.predict(userId1,predictor) 
ret = pred
print(ret)
print("Fin des predictions SpectralCoclustering() pour l 'utilisateur [",userId1,"]")
"""

stop = time.time()

print("Process took ",stop-start," sec")
#beep permet d avoir un signal sonore à la fin de calculs très longs
beep(5)