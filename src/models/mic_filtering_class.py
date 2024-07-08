
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
import sklearn.metrics.pairwise as dist

from scipy.sparse import csr_matrix

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import GridSearchCV
from surprise import SVD

from scipy.spatial.distance import cosine, euclidean, hamming
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns 
import matplotlib.pyplot as plt
import random 

from surprise import SVD
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

import pprint
class mic_base_filter():
    def __init__(self):
        self.data_frame = pd.DataFrame()
        self.df_features = pd.DataFrame()        
        self.artist_key_name = ''        
        self.item_key_name = ''
        self.user_key_name = ''        
        self.visual_key_name = ''
        self.default_user = ''
        self.features = []
        self.target_key_name = ""
        self.file_name = ""
        self.is_content_to_be_filtered = False
        self.is_content_to_be_sparsed = False

        
    #on utilise cette variable pour identifier humainement les mrorceaux
    def init_vars(self):
        self.artist_key_name = ''        
        self.item_key_name = ''
        self.user_key_name = ''        
        self.visual_key_name = ''
        self.default_user = ''
        self.features = []
        self.target_key_name = ""
        
        
    def print_vars(self):
        print("self.artist_key_name = ",self.artist_key_name )
        print("self.item_key_name = ",self.item_key_name )
        print("self.user_key_name = ",self.user_key_name )
        print("self.visual_key_name = ",self.visual_key_name )
        print("self.target_key_name = ",self.target_key_name )
        print("self.features = ",self.features)
        
    def set_visual_key_name(self,key):
        self.visual_key_name = key

    def set_target_key_name(self,key):        
        self.target_key_name = key        
    
    def set_artist_key_name(self,key):
        self.artist_key_name = key

    #def set_track_key_name(self,key):
    def set_item_key_name(self,key):
        self.item_key_name = key

    def set_user_key_name(self,key):
        self.user_key_name = key

    def set_feature_list(self, list):        
        self.features = list

    def get_tracks_with_name(self,name):
        print("************************* get_tracks_of_artist(",name,") **********************")
        
        song_name_lower = name.lower()
        
        df = self.data_frame[self.item_key_name].str.lower() 
        #condition = (df.str.startswith(song_name_lower) )#ok            
        condition = (df.str.contains(song_name_lower) )#ok            
        
        return self.data_frame[condition][[self.artist_key_name,self.item_key_name]]

    def get_artist_closedto(self,artist_name):
        print("type(artist_name) = ",type(artist_name) ,"  artist_name = ",artist_name )
        artist_lower = artist_name.lower()
        df = self.data_frame
        df["artistlower"] = self.data_frame[self.artist_key_name].str.lower() 
        condition = (df['artistlower'].str.contains(artist_lower) )#ok                    

        ldf = df[condition][[self.artist_key_name]]
        ldf.drop_duplicates(subset= [self.artist_key_name], inplace=True)
        print("-----------after drop--",len(ldf))
        retval = []
        for name in ldf[self.artist_key_name].unique():
            
            retval.append(name)
        
        return retval
        

    def get_song_closedto(self,song_name):
        print("        get_song_closedto [",song_name,"] ")
        song_lower = song_name.lower()
        df = self.data_frame
        df["songlower"] = self.data_frame[self.item_key_name].str.lower() 
        condition = (df['songlower'].str.contains(song_lower) )#ok                    

        ldf = df[condition][[self.item_key_name]]
        print(len(df))
        ldf.drop_duplicates(subset= [self.item_key_name], inplace=True)
        print("-----------after drop--",len(ldf))
        retval = []
        for name in ldf[self.item_key_name].unique():
            print("name = ")
            print(name)
            retval.append(name)
        
        return retval
        
    def get_all_artist_songs(self,artist_name):    
        df = self.data_frame[self.data_frame[self.artist_key_name] == artist_name]
        
        sort_mylist = sorted(df[self.item_key_name].to_list(), key=str.lower)
        return sort_mylist 
        #return df[self.item_key_name].to_list()
    

    def get_artist_song_closedto(self,artist_name,song_name):
        print("        get_song_closedto [",song_name,"] ")
        song_lower = song_name.lower()
        df = self.data_frame[self.data_frame[self.artist_key_name] == artist_name]
        df["songlower"] = self.data_frame[self.item_key_name].str.lower() 
        condition = (df['songlower'].str.contains(song_lower) )#ok                    

        ldf = df[condition][[self.item_key_name]]
        print(len(df))
        ldf.drop_duplicates(subset= [self.item_key_name], inplace=True)
        print("-----------after drop--",len(ldf))
        retval = []
        for name in ldf[self.item_key_name].unique():
            print("name = ")
            print(name)
            retval.append(name)
        
        return retval
    

    def get_tracks_of_artist(self,artist_name):
        print("************************* get_tracks_of_artist(",artist_name,") **********************")
        """artist_lower = artist_name.lower()
        df = self.data_frame[self.artist_key_name].str.lower()                     
        condition = (df.str.contains(artist_lower) )#ok                    
        return self.data_frame[condition][[self.artist_key_name,self.item_key_name]]"""
        artist_lower = artist_name.lower()
        ret= pd.DataFrame()
        ret["lower"] = self.data_frame[self.artist_key_name].str.lower() 
        condition = (ret["lower"].str.contains(artist_lower) )#ok            

        ret = ret[condition]    
        return ret
    
    def get_random_track_num_of_artist(self,artist_name):
        ret = self.get_tracks_of_artist(artist_name)        
        n = random.randint(0,len(ret)-1) 
        ret = ret.iloc[n,:]
        return ret

    def get_track_num_of_artist(self,artist_name,song_name):
        print("************************* get_track_num_of_artist(",artist_name,",",song_name,") **********************")
        ret = self.get_tracks_of_artist(artist_name)
        
        song_name_lower = song_name.lower()
        
        ret["lower"] = self.data_frame[self.item_key_name].str.lower() 
        condition = (ret["lower"].str.contains(song_name_lower) )#ok            

        ret = ret[condition]    
        return ret.index.tolist()[0]
    
    def set_data(self,file_name):
        #print("************************* set_data(",file_name,") **********************")
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name)
        
        self.data_frame = self.data_frame.drop_duplicates()
        self.data_frame.style.set_properties(**{'text-align': 'left'})
    
    def get_item_description(self,num):
        return self.data_frame.iloc[num]
    
    def select_by_key_val(self,key , val):
        df = self.data_frame
        df = df[df[key] == val]
        #return self.filter_util_vals(df)
        return df
    
    def select_by_key_approximativ_val(self,key , val):
        df = self.data_frame
        val = val.lower()
        
        df = self.data_frame[key].str.lower() 
        condition = (df.str.contains(val) )#ok            

        return self.filter_util_vals(self.data_frame[condition])
        
        df = df[df[key] == val]
    
        return df
    
    def filter_util_vals(self,df):

        if self.visual_key_name != '':
            ret = df[[self.user_key_name,self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.user_key_name,self.item_key_name,self.target_key_name]]

        return ret    
    

    def filter_util_vals_for_présentation(self,df):

        if self.visual_key_name != '':
            ret = df[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.item_key_name,self.target_key_name]]

        return ret    
        
         
    def get_random_row_from(self,key , val):
        print("*****************get_random_row_from ***************")
        df = self.select_by_key_val(key,val)
        print('len(self.select_by_key_val(',key,',',val,'))',len(df))
        n = random.randint(0,len(df)-1) 
        df = df.iloc[n,:]
        print('self.select_by_key_val(',key,',',val,'))=>\n',df.head())
        
        return self.filter_util_vals(df)
        #return df
    
    def get_random_user_id(self):
        print("**********************get_random_user_id ************************")
        
        n = random.randint(0,len(self.data_frame)-1) 
        #print(n)
        
        df = self.data_frame.iloc[n,:]
        return df[self.user_key_name]
    
    def clean_columns(self):
        print('*****************clean_data() ******************')
        df = self.data_frame
        
        varlist = [self.user_key_name,self.item_key_name,self.target_key_name]
        if self.visual_key_name != '':
            varlist.append( self.visual_key_name)
        #df.info()    
        print(varlist)
        df  = df[varlist]        

        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        df.reindex()        
        self.data_frame = df
                
    def get_user_df(self, userid):
        return self.data_frame[self.data_frame[self.user_key_name] == userid]
    
    def get_user_description(self,userid):
        #print("*************get_user_description(self,userid)***************")
        descrition = 'Description de l utisateur '
        descrition += str(userid)
        df = self.get_user_df(userid)
        if self.visual_key_name != '':
            descrition += '\n'
            descrition += '     Il aime :'
            descrition += str(df.iloc[0][self.visual_key_name ])

        descrition += '\n'
        descrition += '     Il est est pésent ' + str(len(df)) + ' fois dans le dataset'
        return descrition



    def get_random_track_id(self):
        n = random.randint(0,len(self.data_frame)-1) 
        return n

    def get_top_rated_songs(self,num):
        print("**********************get_top_rated_songs ************************")
        df = self.data_frame
        aggregated_data = df.groupby(self.item_key_name )[self.target_key_name].count().reset_index()
        #aggregated_data = df.groupby(self.item_key_name )[self.target_key_name].count()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rated_tracks = sorted_aggregated_data.head(num)
        return top_rated_tracks
    def get_best_voted_songs(self,num):
        
        df = self.data_frame
        # On regroupe les données par titre et calcule le nombre dde notes ('count') et la note moyenne ('mean') pour chaque livre.
        item_stats = df.groupby(self.item_key_name)[self.target_key_name].agg(['count', 'mean']).reset_index()
        item_stats.info()
        print("self.target_key_name = ",self.target_key_name)
        
        # Affichage du graphique
        df = self.data_frame
        aggregated_data = df.groupby(self.item_key_name)[self.target_key_name].mean().reset_index()

        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers tracks les mieux notés
        print("Morceaux les plus mieux notés : ")
        best_rated_tracks = sorted_aggregated_data.head(num)
        print(best_rated_tracks)
        return best_rated_tracks
        #"""
    def get_mean_score_songs(self,num):
        df = self.data_frame

        livre_stats = df.groupby(self.item_key_name)[self.target_key_name].agg(['count', 'mean']).reset_index()

        # On calcule la moyenne du nombre d'avis pour tous les titres.
        C = livre_stats['count'].mean()

        # On calcule la moyenne des notes moyennes pour tous les titres.
        M = livre_stats['mean'].mean()

        # On définit la fonction 'bayesian_avg' qui calcule la note bayésienne pour chaque livre en utilisant les valeurs de C et M calculées précédemment.
        def bayesian_avg(df):
            return (C * M + df.sum()) / (C + df.count())

        # On calcule la note bayésienne pour chaque livre en utilisant la fonction 'bayesian_avg'.
        bayesian_avg_ratings = df.groupby(self.item_key_name)[self.target_key_name].agg(bayesian_avg).reset_index()

        # On renomme les colonnes du DataFrame 'bayesian_avg_ratings' pour les rendre plus explicites.
        #bayesian_avg_ratings.columns = [self.item_key_name, 'bayesian_avg']
        bayesian_avg_ratings.columns = [self.item_key_name, self.target_key_name]

        # On fusionne 'livre_stats' avec les moyennes bayésiennes en utilisant le titre comme clé et on tri par moyenne bayesienne en ordre décroissant.
        #track_stats = livre_stats.merge(bayesian_avg_ratings, on=self.item_key_name).sort_values('bayesian_avg', ascending=False)
        track_stats = livre_stats.merge(bayesian_avg_ratings, on=self.item_key_name).sort_values(self.target_key_name, ascending=False)

        ## Sélection des 10 premiers livres les mieux notés
        #best_rated_tracks = track_stats[[self.item_key_name, 'bayesian_avg']].head(num)
        best_rated_tracks = track_stats[[self.item_key_name, self.target_key_name]].head(num)
        return best_rated_tracks

    
    def get_top_rater_users(self,num):
        df = self.data_frame
        aggregated_data = df.groupby(self.user_key_name )[self.target_key_name].count().reset_index()
        
        # Tri du DataFrame agrégée par note en ordre décroissant
        sorted_aggregated_data = aggregated_data.sort_values(by=self.target_key_name, ascending=False)

        # Sélection des 10 premiers morceaux les mieux notés
        top_rater_users = sorted_aggregated_data.head(num)
        return top_rater_users

    #normalisation des donées de sortie
    def get_full_data(self,indf,key ):
        #print("***************** get_full_data ******************")
        ret = self.data_frame.merge(indf,how='right',on=key )
        if self.visual_key_name == '':
            ret = ret[self.item_key_name]
        else:
            ret = ret[[self.item_key_name,self.visual_key_name]]
        
        return ret
    
    def get_util_data_for_presentation(self,df):
        #print("***************** get_util_data_for_presentation ******************")
        ret = any
        if self.visual_key_name != '':
            ret = df[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :
            ret = df[[self.item_key_name,self.target_key_name]]
        
        return ret
        
    def drop_dupicated_tracks(self):
        print("********************* drop_dupicated_tracks *************")
        df = self.data_frame
        print(" before drop_duplicates shape = ",df.shape)
        print(df.head(10))
        if self.user_key_name == '':
              df = df.drop_duplicates(subset=[self.item_key_name,self.artist_key_name])  
        else:
              df = df.drop_duplicates(subset=[self.item_key_name,self.artist_key_name,self.user_key_name])  
        df = df.reset_index(drop=True)
        self.data_frame = df
        print(" after drop_duplicates shape = ",df.shape)
        print(df.head(10))
        
    

    def get_best_voter_users(self,num):    
        df = self.data_frame
        df = self.data_frame
        agg_dict = {self.item_key_name:lambda x:list(x)}

        df = df[df[self.target_key_name] != np.nan]

        big_user = df.groupby(self.user_key_name).agg(agg_dict).reset_index()
        big_user['count'] = big_user[self.item_key_name].str.len()
        
        big_user = big_user.drop([self.item_key_name],axis=1)        

        big_user = big_user.sort_values(by='count',ascending=False)
        user = big_user[self.user_key_name].iloc[0]

        return big_user.head(num)

    def display_tracks_and_users_num(self):
        n_users = self.data_frame[self.user_key_name].nunique()

        n_tracks = self.data_frame[self.item_key_name].nunique()

        print("Nombre d 'utilisateurs : ",n_users)
        print("Nombre de morceaux : ",n_tracks)

    #def get_df_for_heatmap(self):
    def get_numerical_var(self):
        
        df = self.data_frame
        name_list_to_keep = []
        for val in df.columns:
            #print('get_df_for_heatmap check column ',val)
            if is_numeric_dtype(df[val]) :
                #print(" colonne[",val,"] est numerique")
                name_list_to_keep . append(val)
        #print(name_list_to_keep)
        #df.info()
        return df[name_list_to_keep]
    
    def get_nan_in_column(self,key):
        df = self.data_frame
        nb = df[key].isna().sum()
        print("get_nan_in_column(",key,") => ",nb)
        return nb

    def replace_nan_by_zero(self,key):
        print(" replace_nan_by_zero():")
        self.data_frame[key].fillna(0,inplace=True)
            
    def replace_nan_by_mean(self,key):
        print("replace_nan_mean()")
        self.data_frame[key].fillna(self.data_frame[key].mean(),inplace=True)        

    def remove_nan(self,key):
        print("remove_nan()")
        self.data_frame[key].dropna(inplace=True)


#########################################################""    
        
class mic_content_filtering( mic_base_filter):
    """
    mic_content_filtering
    """
    def __init__(self):
        mic_base_filter.__init__(self)
        self.df_normalized = pd.DataFrame()

    def normalize_data(self)    :        
        #print('*********************** normalize_data ******************')
        
        print("normalze_data(self): before self.data_frame.drop_duplicates shape = ",self.data_frame.shape)
        llist = self.features.copy()
        llist .append(self.artist_key_name)
        llist .append(self.item_key_name)        
        self.data_frame.drop_duplicates(subset  = llist,keep='first' ,inplace=True)
        print("normalize_data(self): after self.data_frame.drop_duplicates shape = ",self.data_frame.shape)

        self.data_frame[self.artist_key_name] = self.data_frame[self.artist_key_name].map(lambda x: x.lstrip('[').rstrip(']'))

        self.data_frame[self.artist_key_name] = self.data_frame[self.artist_key_name].map(lambda x: x[1:-1])
        
        self.data_frame['song_id']=self.data_frame.index

        self.df_normalized = self.data_frame[self.features]
        
        #self.df_normalized.index = self.data_frame['song_id']
        self.df_normalized = pd.DataFrame(normalize(self.df_normalized, axis=1))
        #from sklearn.preprocessing import StandardScaler
        #sc = StandardScaler()
        #Z = sc.fit_transform(dfwithoutname)

        self.df_normalized.index = self.data_frame['song_id']

        



    def content_filter_music_recommender(self,songidp, N):
        print("***************** content_filter_music_recommender(songidp = ",songidp,") *************************")
        
        distance_method = euclidean
        #distance_method =  hamming
        
        print(self.data_frame[self.artist_key_name].head(10))
                
        allSongs = pd.DataFrame(self.df_normalized.index)
        allSongs["distance"] = allSongs["song_id"].apply(lambda x: distance_method(self.df_normalized.loc[songidp], self.df_normalized.loc[x]))
        # sort by distance then recipe id, the smaller value of recipe id will be picked. 
        TopNRecommendation = allSongs.sort_values(["distance"]).head(N).sort_values(by=['distance', 'song_id'])
        Recommendation = pd.merge(TopNRecommendation , self.data_frame, how='inner', on='song_id')        
        
        TopNUnRecommendation = allSongs.sort_values(["distance"]).tail(N).sort_values(by=['distance', 'song_id'])
        #UnRecommendation = pd.merge(TopNUnRecommendation , self.data_frame, how='inner', on='song_id')
        #print("les plus éloignés....", UnRecommendation.tail(N))
        
        SongName = Recommendation[self.item_key_name ]  
        ArtisName = Recommendation[self.artist_key_name ]
        Distance = Recommendation["distance"]
        SongId  = Recommendation["song_id"]

        return SongId , SongName , ArtisName , Distance


########################################

class mic_collaborativ_filtering(mic_base_filter ):
    """
    mic_collaborativ_filtering
    """
    def __init__(self):
        print("***************** __init()__ *************************")
        mic_base_filter.__init__(self)
        #self.data_frame = pd.DataFrame()
        self.matrice_pivot = pd.DataFrame()
        self.sparse_ratings = any
        self.track_ids = list()
        self.user_ids = list()
        

    def generate_notation_mattrix(self):
        print("*************** generate_notation_mattrix ******************")
        df = self.data_frame

                
        n_users = df[self.user_key_name].nunique()

        #n_tracks = df[self.item_key].nunique()
        n_tracks = df[self.item_key_name].nunique()

        print("Nombre d 'utilisateurs : ",n_users)
        print("Nombre de morceaux : ",n_tracks)
        #la matrice de notations 
        #               chaque ligne représente les notes données par un utilisateur 
        #              chaque colonne les notes attribuées à un contenu. 
        #self.matrice_pivot = df.pivot_table(columns=self.item_key, index=self.user_key_name, values=self.target_key_name)
        
        #data = np.array(df)
        #data = np.asmatrix(df,colu)
        #data = np.as_matrix(df)
        #print(data)
        self.matrice_pivot = df.pivot_table(columns=self.item_key_name
                                            , index=self.user_key_name
                                            , values=self.target_key_name)

        #print(self.matrice_pivot)
        print("Nombre de notes manquantes : ",self.matrice_pivot.isna().value_counts().sum())

        #25.06.2024
        self.matrice_pivot = self.matrice_pivot +1
        
        self.matrice_pivot.fillna(0, inplace=True)
        #

        print("Matrice de notation (shape = ",self.matrice_pivot.shape,") \n\
              ---les colonnes contiennent les notes données par un utilisateur\n\
              ---les lignes contiennent les notes attribuées à un contenu")
        
        # Convertir la matrice de notations 'self.matrice_pivot' en une matrice creuse 'sparse_ratings'.
        self.sparse_ratings = csr_matrix(self.matrice_pivot)        
        #print(self.sparse_ratings)
        """
        note
            content--        c1                  c2             c3              ... 
        user    
          |

        u1                  note[u1,c1]         none            note[u1,c3]     ...
        u2                  none                note[u2,c2]     none            ...
        u3                  note[u3,c1]         note[u3,c2]     note[u3,c3]     ...
        ...
        """

        # Extraire les identifiants des utilisateurs et les track_id à partir de la matrice de notations.
        self.user_ids = self.matrice_pivot.index.tolist()  
        #print("Nombre de users dans la matrice pivot : ",len(self.user_ids) )
        self.track_ids = self.matrice_pivot.columns.tolist()  
        #print("Nombre de morceaux dans la matrice pivot : ",len(self.track_ids) )

        self.user_similarity = self.get_user_similarity()
        self.item_similarity = self.get_item_similarity()

        # Afficher la matrice creuse 'sparse_ratings'.
        #print(sparse_ratings)

    def get_users_similarity(self,user_id1,user_id2):
        #  Insérez votre réponse ici 
        pref_1 = self.matrice_pivot.loc[user_id1, :].values

        pref_2 = self.matrice_pivot.loc[user_id2, :].values

        similarity = self.sim_cos(pref_1, pref_2)

        print("La similarité entre les deux utilisateurs est ", similarity)
        return similarity

    def get_preference(self,user_id):
        return self.matrice_pivot.loc[user_id, :].values    

    def sim_cos(self,x, y):
        # Calcul du produit scalaire entre les vecteurs 'x' et 'y'.
        dot_product = np.dot(x, y)
        
        # Calcul des normes euclidiennes de 'x' et 'y'.
        norm_x = np.sqrt(np.sum(x ** 2))
        norm_y = np.sqrt(np.sum(y ** 2))
        
        # Vérification si l'une des normes est nulle pour éviter une division par zéro.
        if norm_x == 0 or norm_y == 0:
            return 0
        
        # Calcul de la similarité cosinus en utilisant la formule.
        similarity = dot_product / (norm_x * norm_y)
        return similarity
    
    def  get_preferences(self,userid,score_seuil,number_of_ligns = None) -> pd.DataFrame :                
        
        
        user_preferences = self.data_frame[(self.data_frame[self.user_key_name] == userid) 
                                           & (self.data_frame[self.target_key_name] >= score_seuil)
                                           & (self.data_frame[self.target_key_name] != np.nan)]        
        
        
        user_preferences = user_preferences.sort_values(self.target_key_name, ascending=False).drop_duplicates(subset=[self.item_key_name])        
                        
        if number_of_ligns != None:
            user_preferences = user_preferences.head(number_of_ligns)            

        ret = user_preferences
        if self.visual_key_name != '':
            ret = ret[[self.item_key_name,self.visual_key_name,self.target_key_name]]
        else :    
            ret = ret[[self.item_key_name,self.target_key_name]]    
        ret = ret.sort_values(self.target_key_name, ascending=False)
        

        #print("**************************** get_preferences(score_seuil = ",score_seuil,") ****************************>")
        return ret
    
    def get_item_similarity(self):
        #print("************** get_item_similarity *************************")
        item_similarity = dist.cosine_similarity(self.sparse_ratings.T)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        item_similarity = pd.DataFrame(item_similarity, index=self.track_ids, columns=self.track_ids)
        
        return item_similarity

    def get_user_similarity(self):
        #print("************** get_user_similarity *************************")   
        # Utilisation de la fonction 'cosine_similarity' du module 'dist' pour calculer la similarité cosinus entre les utilisateurs.    
        user_similarity = dist.cosine_similarity(self.sparse_ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        user_similarity = pd.DataFrame(user_similarity, index=self.user_ids, columns=self.user_ids)
        
        return user_similarity
    
    def get_similar_users(self,  nearest_count, user_id):
        
        #user_similarity = self.get_user_similarity()
        # Utilisation de la fonction 'cosine_similarity' du module 'dist' pour calculer la similarité cosinus entre les utilisateurs.
        user_similarity = dist.cosine_similarity(self.sparse_ratings)

        # Création d'un DataFrame pandas à partir de la matrice de similarité entre utilisateurs.
        # Les index et les colonnes du DataFrame sont les identifiants des utilisateurs.
        user_similarity = pd.DataFrame(user_similarity, index=self.user_ids, columns=self.user_ids)
        
        # Sélectionner dans matrice pivot les morceaux qui n'ont pas été encore écouté par le user        

        # Sélectionner les k users les plus similaires en excluant le user lui-même
        similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
        print("type(similar_users) = ",type(similar_users))
        return similar_users
        
        
    
    def pred_user(self,  nearest_count, user_id,number_of_predictions=None):
        print("*************** pred_user *****************")


        # Sélectionner dans mat_ratings les contenus qui n'ont pas été encore écouté par le user
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]                
            
        # Sélectionner les k users les plus similaires en excluant le user lui-même
        #similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
        similar_users = self.get_similar_users(nearest_count,user_id)

##################################################################################################
        #user_similarity = dist.cosine_similarity(self.sparse_ratings)
        
        
        #user_similarity = pd.DataFrame(user_similarity, index=self.user_ids, columns=self.user_ids)
##################################################################################################        

        #similar_users = user_similarity.loc[user_id].sort_values(ascending=False)[1:nearest_count+1]
                
        # Calcul du dénominateur
        norm = np.sum(np.abs(similar_users))
        print("len(to_predict.index) = ",len(to_predict.index))
        
        for i in to_predict.index:
            # Récupérer les notes des users similaires associées au morceau i
            ratings = self.matrice_pivot[i].loc[similar_users.index]
            # Calculer le produit scalaire entre ratings et similar_users
            scalar_prod = np.dot(ratings,similar_users)            
            #Calculer la note prédite pour le film i
            pred = scalar_prod / norm

            # Remplacer par la prédiction

            to_predict[i] = pred
            #if pred != 0:
            #    print("to_predict[",i,"]" , pred)
        retpredict = to_predict
        print("len(retpredict)",len(retpredict))
        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        #to_predict = to_predict.sort_values(by=self.target_key_name,ascending =False)
        
        #retpredict = retpredict[retpredict != 0]    

        retpredict = retpredict.sort_values(ascending =False)

        if number_of_predictions != None:
            retpredict  = retpredict.head(number_of_predictions )



        return retpredict

    #def pred_item(mat_ratings, item_similarity, k, user_id):
    def pred_item(self, nearest_count, user_id,number_of_predictions=None):
        print("*************** pred_item user_id[",user_id,"]*****************")
        item_similarity = self.get_item_similarity()
        
        # Sélectionner dans la self.matrice_pivot les morcecaux qui n'ont pas été encore écouté par l utilisateur
        #print(self.matrice_pivot.loc[user_id])
        
        to_predict = self.matrice_pivot.loc[user_id][self.matrice_pivot.loc[user_id]==0]
        #to_predict.rename("score moyen", inplace=True)
        """
        for i in to_predict.index:            
            similar_items = item_similarity.loc[i].sort_values(ascending=False)[1:nearest_count+1]            
            norm = np.sum(np.abs(similar_items))            
            ratings = self.matrice_pivot[similar_items.index].loc[user_id]
            scalar_prod = np.dot(ratings,similar_items)                        
            pred = scalar_prod / norm                    
            to_predict[i] = pred
        """
        
        # Itérer sur tous ces morceaux 
        for i in to_predict.index:

            #Trouver les k morceaux les plus similaires en excluant le morceau lui-même
            similar_items = item_similarity.loc[i].sort_values(ascending=False)[1:nearest_count+1]

            # Calcul de la norme du vecteur similar_items
            norm = np.sum(np.abs(similar_items))

            # Récupérer les notes données par l'utilisateur aux k plus proches voisins
            ratings = self.matrice_pivot[similar_items.index].loc[user_id]

            # Calculer le produit scalaire entre ratings et similar_items
            scalar_prod = np.dot(ratings,similar_items)
            
            #Calculer la note prédite pour le morceau i
            pred = scalar_prod / norm
        
            # Remplacer par la prédiction
            if pred != 0:
                to_predict[i] = pred
        
        #to_predict = pd.merge(to_predict,self.data_frame, on=[self.item_key], how='inner')
        
        #to_predict = to_predict[to_predict != 0]
        to_predict = to_predict.sort_values(ascending =False)
        
        if number_of_predictions != None:
            to_predict  = to_predict.head(number_of_predictions )
        
        return to_predict

#################################################"modéle hybride"
    
class mic_hybrid_filtering(mic_base_filter):
    def __init__(self):
        #print('*****************mic_hybrid_filtering __init()__ *************************')
        mic_base_filter.__init__(self)
        #self.best_params_predictor = {}
        #self.best_scor ={}
        #self.average_mae = 0
        #self.average_rmse = 0

        self.data = any
        
    def set_data(self, file_name):
        super().set_data(file_name)
        self.print_vars()
        
        self.generate_dateset_autofold()
        """min_rating = self.data_frame[self.target_key_name].min()
        max_rating = self.data_frame[self.target_key_name].max()

        print("min_rating = ",min_rating)
        print("max_rating = ",max_rating)

        reader = Reader(rating_scale=(min_rating, max_rating))
        self.data = Dataset.load_from_df(self.data_frame[[self.user_key_name,
                                                                 self.item_key_name,
                                                                  self.target_key_name]], reader)
        print(self.data_frame.dtypes)"""
        #exit()
    def get_and_remove_from_dataframe(self , userId,trackid):
        df = self.data_frame.copy()
        condition = (df[self.user_key_name] == userId) & (df[self.item_key_name] == trackid)
        indextodrop = df[condition].index
                
        self.data_frame.drop(indextodrop,inplace=True)

        return df[condition]

    def generate_dateset_autofold(self):
        self.min_rating = self.data_frame[self.target_key_name].min()
        self.max_rating = self.data_frame[self.target_key_name].max()

        print("min_rating = ",self.min_rating)
        print("max_rating = ",self.max_rating)

        reader = Reader(rating_scale=(self.min_rating, self.max_rating))
        self.data = Dataset.load_from_df(self.data_frame[[self.user_key_name,
                                                                 self.item_key_name,
                                                                  self.target_key_name]], reader)
    def evaluate_predictor(self,predictor):
        print("-------------evaluate_predictor(self,predictor):")
        
        ret = cross_validate(predictor, self.data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        print("evaluate_predictor => ")
        #print(type(ret))
        #print(ret)
        pprint.pprint(ret, width=4)
        print("predictor_ajustement() Average MAE: ", np.average(ret["test_mae"]))
        print("predictor_ajustement() Average RMSE: ", np.average(ret["test_rmse"]))
        self.average_eveluation_mae = np.average(ret["test_mae"])
        self.average_eveluation_rmse = np.average(ret["test_rmse"])
        
    def evaluate_predictions(self,predictions):
        from surprise import accuracy
        self.mae_model_predictions = accuracy.mae(predictions)
        self.rmse_model_predictions = accuracy.rmse(predictions)
        print("evaluate_predictions(self,predictions): self.mae_model_predictions = ",self.mae_model_predictions)
        print("evaluate_predictions(self,predictions): self.rmse_model_predictions = ",self.rmse_model_predictions)
        

    def predictor_ajustement(self,predictor,param_search):    
        print("-------------predictor_ajustement(self,predictor,param_search):")
        
        
        gs = GridSearchCV(predictor, param_search, measures=['rmse', 'mae'], cv=10)
        gs.fit(self.data)
        print("GridSearch Best score =>")
        pprint.pprint(gs.best_score, width=4)
        print("GridSearch Best params =>")
        pprint.pprint(gs.best_params, width=4)
        self.best_params_predictor = gs.best_params
        self.best_score = gs.best_score
        #self.evaluate_predictor(predictor)
        return self.best_params_predictor

    """def compute_antitraintest1(self,user_id):        
        print("---------------------------------------------------------compute_antitraintest1(self,user_id):")
        df = self.data_frame
        min_rating = self.data_frame[self.target_key_name].min()
        max_rating = self.data_frame[self.target_key_name].max()

        reader = Reader(rating_scale=(min_rating, max_rating))
        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)
        # Construire le jeu d'entraînement complet à partir du DataFrame df_surprise
        train_set = df_surprise.build_full_trainset()
        
        # Convertir l'ID de l'utilisateur externe en l'ID interne utilisé par Surpsultsrise
        targetUser = train_set.to_inner_uid(user_id)

        # Obtenir la valeur de remplissage à utiliser (moyenne globale des notes du jeu d'entraînement)
        moyenne = train_set.global_mean

        # Obtenir les évaluations de l'utilisateur cible pour les tracks
        user_note = train_set.ur[targetUser]

        # Extraire la liste des livres notés par l'utilisateur
        user_track = [item for (item,_) in (user_note)]

        # Obtenir toutes les notations du jeu d'entraînement
        ratings = train_set.all_ratings()

        # Initialiser une liste vide pour stocker les paires (utilisateur, livre) pour le jeu "anti-testset"
        anti_testset = []
        testset = []
        # Boucle sur tous les items du jeu d'entraînement
        for livre in train_set.all_items():
            # Si l'item n'a pas été noté par l'utilisateur
            if livre not in user_track:
                # Ajouter la paire (utilisateur, livre, valeur de remplissage) à la liste "anti-testset"
                anti_testset.append((user_id, train_set.to_raw_iid(livre), moyenne))
            else:
                testset.append((user_id, train_set.to_raw_iid(livre), moyenne))                
        print("l'utilisateur [",user_id,"] a écouté ",len(testset)," morceaux")
        print("l'utilisateur [",user_id,"] n a pas écouté ",len(anti_testset)," morceaux")
        print("track_ids_to_pred = \n",anti_testset[0:10])

        return anti_testset"""



    def compute_traintest(self,user_id):
        print("---------------------------------------------------------compute_antitraintest(self,user_id):")
                # Récupération de la liste des morceaux
        track_ids = self.data_frame[self.item_key_name].unique()
        
        #Récupération des morceaux écoutés par l'utilisateur identifié par user_id
        #track_ids_user = self.data_frame.loc[self.data_frame[self.user_key_name] == user_id, self.item_key_name]
        track_ids_user = self.data_frame.loc[self.data_frame[self.user_key_name] == user_id]

        #On considère que les morceaux sans note n'on pas étét écoutés
        #track_ids_user = track_ids_user.dropna()

        track_ids_user = track_ids_user.loc[self.data_frame[self.user_key_name] == user_id, self.item_key_name]
        #track_ids_user = track_ids_user
        print("l'utilisateur [",user_id,"] a écouté ",len(track_ids_user)," morceaux")
        
        # Récupération des morceaux non écoutés par l'utilisateur identifié par user_id
        #track_ids_to_pred = np.setdiff1d(track_ids, track_ids_user)        
            #track_ids_to_pred = self.data_frame.loc[self.data_frame[self.user_key_name] != user_id, self.item_key_name]

        track_ids_to_pred = track_ids_user
        print("l'utilisateur [",user_id,"] n a pas écouté ",len(track_ids_to_pred)," morceaux")
        print("track_ids_to_pred = \n",track_ids_to_pred[0:10])
        #print(track_ids_to_pred[0])
                    
        list_out = [[user_id, track_id, 0] for track_id in track_ids_to_pred]
        print("l'utilisateur [",user_id,"] len(list_out) ",len(list_out)," morceaux")
        print("list_out = \n",list_out[0:10])
        return list_out
 

    def compute_antitraintest_with_threshold(self,user_id,seuil ):
        print("compute_antitraintest_with_threshold")
        df = self.data_frame
        max_rating = self.data_frame[self.target_key_name].max()
        min_rating = self.data_frame[self.target_key_name].min()

        reader = Reader(rating_scale=(min_rating, max_rating))
        df_surprise = Dataset.load_from_df(df[[self.user_key_name, self.item_key_name, self.target_key_name]], reader=reader)
        
        train_set = df_surprise.build_full_trainset()

        targetUser = train_set.to_inner_uid(user_id)        
        
        # Obtenir la valeur de remplissage à utiliser (moyenne globale des notes du jeu d'entraînement)
        moyenne = train_set.global_mean
        print("moyenne = ",moyenne)
        
        # Obtenir les évaluations de l'utilisateur cible pour les trackss
        user_note = train_set.ur[targetUser]
        #print(user_note)
        #on enlève les morceau non  notés, si pas noté pas écouté...
        from math import isnan
        #suppression des nan de la liste de tuple
        #user_note = [t for t in user_note if not any(isinstance(n, float) and isnan(n) for n in t)]        
        user_note = [t for t in user_note 
                     if not any( isnan(n)  for n in t) 
                     #and any(isinstance(n, float) and n > seuil for n in t)]        
                     and any(isinstance(n, float) and n > seuil for n in t)]        
        
        # Extraire la liste des morceaux notés par l'utilisateur
        user_tracks = [item for (item,_) in (user_note)]

        # Obtenir toutes les notations du jeu d'entraînement
        #ratings = train_set.all_ratings()
        #print()
        #for el in ratings:
        #    print(el)
        
        list_out = []
        """
        for track , rating in zip(train_set.all_items(),train_set.all_ratings()):
            # Si l'item n'a pas été noté par l'utilisateur
            if track not in user_tracks:
                # Ajouter la paire (utilisateur, morceau, valeur de remplissage) à la liste "anti-testset"
                list_out.append((user_id, train_set.to_raw_iid(track), rating))
        """                
        # Boucle sur tous les items du jeu d'entraînement
        for track in train_set.all_items():
            # Si l'item n'a pas été noté par l'utilisateur
            if track not in user_tracks:
                # Ajouter la paire (utilisateur, morceau, valeur de remplissage) à la liste "anti-testset"
                list_out.append((user_id, train_set.to_raw_iid(track), moyenne))
        
        return list_out
    
    def compute_trainset(self):        
        print("compute_trainset(self): ")
        self.trainset = self.data.build_full_trainset()
        
        
    """def compute_testset(self):        
        print("compute_testset(self): ")
        self.testset = self.compute_antitraintest_with_threshold(user_id,0)"""

    def predictwith_testset(self,testp,predictor,number_of_predictions=None):
        trainset = self.data.build_full_trainset()

        predictions = predictor.fit(trainset).test(testp)

        #ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        #predictions = predictions.rename(ldict,axis=1)
        #if number_of_predictions != None:
        #    predictions  = predictions.head(number_of_predictions )
        return predictions
        return 

    #pred = algo.predict(uid, iid, r_ui=4, verbose=True)
    def predict(self,user_id,predictor,number_of_predictions=None) :        
        print("************predict *********************")      
        trainset = self.data.build_full_trainset()
        print('trainset.n_users = ',trainset.n_users)
        print('trainset.n_items = ',trainset.n_items)
        print('trainset.n_ratings = ',trainset.n_ratings)
        testset = self.compute_antitraintest_with_threshold(user_id,0)
        print('len(testset) = ',len(testset))
        # Prediction des scores and generations des recommendations
        predictions = predictor.fit(trainset).test(testset)
        predictions = pd.DataFrame(predictions)

        predictions.sort_values(by=['est'], inplace=True, ascending=False)

        ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        predictions = predictions.rename(ldict,axis=1)
        if number_of_predictions != None:
            predictions  = predictions.head(number_of_predictions )
        return predictions

    def predict_with_train_split(self,user_id,predictor,number_of_predictions=None) :      
        print("************predict_with_train_split *********************")              
        trainset, testset = train_test_split(self.data, test_size=.50)
        
        print('trainset.n_users = ',trainset.n_users)
        print('trainset.n_items = ',trainset.n_items)
        print('trainset.n_ratings = ',trainset.n_ratings)

        #testset = self.compute_antitraintest(user_id)
        testset = self.compute_antitraintest_with_threshold(user_id,4)
        #build_testset
        print('len(testset) = ',len(testset))
        #############################    
        predictor.fit(trainset)
        
        # Construire le jeu d'entraînement complet à partir du DataFrame df_surprise        
        
        predictions = predictor.test(testset)  
        #
        predictions = pd.DataFrame(predictions)

        # Trier les prédictions par estmiations décroissantes
        predictions.sort_values(by=['est'], inplace=True, ascending=False)
        
        ldict ={'uid':self.user_key_name,'iid':self.item_key_name}
        predictions = predictions.rename(ldict,axis=1)
        if number_of_predictions != None:
            predictions  = predictions.head(number_of_predictions )
        return predictions 

    