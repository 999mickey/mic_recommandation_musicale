#print("visualize.py =>__package__ ",__package__ )
#print("visualize.py =>__name__ ",__name__ )
#print("visualize.py =>__file__ ",__file__ )

from models.mic_filtering_class import mic_base_filter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

import matplotlib.image as mpimg
class mic_vizualizer(mic_base_filter ):
    """
    mic_vizualizer
    objet de vizualisation de dataset
    """
    def __init__(self):
         mic_base_filter.__init__(self)


    def plot_repartion_cat_target(self,key_name,dump = None):
        print("************** plot_repartion_cat_target(",key_name,") **********************")

        filename = f'png/{self.file_name}plot_repartion_cat_target{key}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                        
            df = self.data_frame        
            
            max =  df[key_name].nunique()
            print("             max = ",max)
            
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.countplot(x=key_name, data=df, palette="viridis",bins=10)
            #fig,ax = plt.subplots(figsize=(12,4))
            #ax.set_ylim(0, max)    

            #axes.set_xticks(axes.get_xticks()[::2]
            if dump == True:
                plt.savefig('hist_'+key_name+'1.png')
            plt.show()
            plt.savefig(filename)
        return fig
        #plt.close()


    def plot_repartion_count_target(self,key_name,dump = None):
        print("************** plot_repartion_cont_target **********************")
        filename = f'png/{self.file_name}plot_repartion_count_target{key_name}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            plt.imshow(img)
            
        else :                        
            df = self.data_frame        
            
            max =  df[key_name].max()
            print("             max = ",max)
            fig, axes = plt.subplots(figsize=(10, 10))
            
            #sns.countplot(x=key_name, data=df, palette="viridis",bins=10)
            sns.displot(x=key_name, data=df, palette="viridis",bins=10)
            plt.savefig(filename)
            #sns.displot(df['age'],bins=10,rug=True,color = 'red',kde=True)
            #fig,ax = plt.subplots(figsize=(12,4))
            #ax.set_ylim(0, max)

            #axes.set_xticks(axes.get_xticks()[::2]
            #plt.show()
            #plt.close()
        return fig

    """
    def plot_repartion_target(self,dump = None):
        print("************** plot_repartion_target **********************")
        df = self.data_frame        
        sns.countplot(x=self.target_key_name, data=df, palette="viridis")
        plt.title("Distribution des sentimentss", fontsize=14)
        
        #axes.set_xticks(axes.get_xticks()[::2]
        plt.savefig('matrice_de_notation_user_track_sentiment.png')
        plt.show()
        plt.close()
    """
    def plot_most_popular_tracks(self,num,dump = None):
        print("***************** plot_most_popular_tracks ********************")

        filename = f'png/{self.file_name}plot_most_popular_tracks{num}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                        
            df = self.data_frame                        
            sorted_aggregated_data = self.get_top_rated_songs(num)
            # Sélection des 10 premiers morceaux les plus populaires
            print("Morceaux les plus populaires : ")
            top_rated_tracks = sorted_aggregated_data.head(num)
            print(top_rated_tracks)
            #df = df.set_index(self.item_key_name).T
            
            fig, axes = plt.subplots(figsize=(10, 10))
            plt.yticks(rotation=45)
            plt.yticks(fontsize=6)
            plt.title(f'Top '+str(num)+' track les plus Populaires')
            if dump == True:
                plt.savefig('hist_tracks_popularity_'+str(num)+'.png')            
            fig, ax = plt.subplots()
            sns.barplot(y=self.item_key_name, 
                        x=self.target_key_name, 
                        data=top_rated_tracks, orient = 'h',
                        color = 'red', hue=self.target_key_name)
            plt.savefig(filename)
        return fig


    def plot_best_noted_tracks(self,num,dump = None):
        print("***************** plot_best_noted_tracks ********************")
        filename = f'png/{self.file_name}plot_best_noted_tracks{num}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            plt.imshow(img)
            
        else :                        
            
            best_rated_tracks = self.get_best_voted_songs(num)        

            print("  len(best_rated_tracks) = ",len(best_rated_tracks))
                    
            #ax.tick_params(direction='out', length=6, width=2, colors='r',grid_color='r', grid_alpha=0.5)
            figure(figsize=(10, 6), dpi=120)
            plt.yticks(rotation=45)
            plt.yticks(fontsize=6)
            plt.title(f'Top '+str(num)+' tracks les mieux notés')
            if dump == True:
                plt.savefig('hist_tracks_bestnoted_'+str(num)+'.png')                    
            plt.xlabel("Note moyenne")
            # Affichage du graphique
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.barplot(y=self.item_key_name,
                        x=self.target_key_name,
                        data=best_rated_tracks,
                            orient = 'h',
                            color = 'red', hue=self.target_key_name)
            plt.savefig(filename)
        
        return fig
    

    def plot_best_mean_noted_tracks(self,num,dump = None):
        print("***************** plot_best_noted_tracks ********************")
        filename = f'png/{self.file_name}plot_best_mean_noted_tracks{num}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                        

            best_rated_tracks = self.get_mean_score_songs(num)        

            print("  len(best_rated_tracks) = ",len(best_rated_tracks))
                    
            #ax.tick_params(direction='out', length=6, width=2, colors='r',grid_color='r', grid_alpha=0.5)
            figure(figsize=(10, 6), dpi=120)
            plt.yticks(rotation=45)
            plt.yticks(fontsize=6)
            plt.title(f'Top '+str(num)+' tracks les mieux notés')
            if dump == True:
                plt.savefig('plot_best_mean_'+str(num)+'.png')                    
            plt.xlabel("Note moyenne")
            # Affichage du graphique
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.barplot(y=self.item_key_name,
                        x=self.target_key_name,
                        data=best_rated_tracks,                       
                            orient = 'h',
                            color = 'red', hue=self.target_key_name)
            plt.savefig(filename)

        return fig



    def plot_most_voter_user(self,num,dump = None):
        print("***************** plot_most_voter_user ********************")
        filename = f'png/{self.file_name}plot_most_voter_user{num}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                        
            
            gb_user = self.get_best_voter_users(num)
            print(gb_user)

            # Affichage du graphique
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.barplot(y=self.user_key_name, x='count', data=gb_user, orient = 'h')
            plt.xlabel('Nombre de votes')
            plt.ylabel('User')
            plt.title(f'Top ',str(num),' des user les plus participatifs')
            
            plt.savefig(filename)                    
                
        return fig

    def plot_heatmap(self):

        filename = f'png/{self.file_name}heatmap.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                        
            
            dfwithoutname = self.get_numerical_var()        
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.heatmap(dfwithoutname.corr(), ax=axes)
            plt.savefig(filename)
        return fig

    def plot_repartion_count_target(self,key_name,dump = None):
        print("************** plot_repartion_cont_target **********************")
        filename = f'png/{self.file_name}plot_repartion_count_target{key_name}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")
            fig, axes = plt.subplots(figsize=(10, 10))
            img = mpimg.imread(filename)
            
            plt.imshow(img)
            
        else :                                
            df = self.data_frame        
            
            fig, axes = plt.subplots(figsize=(10, 10))
            sns.countplot(x=key_name, data=df,palette="dark:red_r")
            plt.savefig(filename)
        return fig

    def plot_principal_components(self):
        filename = f'png/{self.file_name}plot_principal_components{key_name}.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")

            img = mpimg.imread(filename)
            fig, axes = plt.subplots(figsize=(10, 10))
            plt.imshow(img)
            
        else :                        
            pca = PCA()
            dfwithoutname = self.get_numerical_var()        
            sc = StandardScaler()
            Z = sc.fit_transform(dfwithoutname)
            fig, axes = plt.subplots(figsize=(10, 10))
            Coord = pca.fit_transform(Z)
            print(pca.explained_variance_)
            print(len(pca.explained_variance_))        
            fig = plt.plot(range(0,len(pca.explained_variance_)), pca.explained_variance_)
            
            plt.xlabel('Nombre de facteurs')
            plt.ylabel('Valeurs propres')
            #filename = 'png/{self.file_name}plot_principal_components.png'
            #plt.savefig(filename)
            #plt.show()
            plt.savefig(filename)
        return fig

    def plot_pie_of_principal_comp(self):
        print("***************************************plot_pie_of_principal_comp")
        filename = f'png/{self.file_name}plot_pie_of_principal_comp.png'
        file_exists = os.path.exists(filename)

        if file_exists :
            print("-------------- file exists")

            img = mpimg.imread(filename)
            fig, axes = plt.subplots(figsize=(10, 10))
            plt.imshow(img)
            
        else :                        
    
            pca = PCA()
            dfwithoutname = self.get_numerical_var()        
            sc = StandardScaler()
            fig, axes = plt.subplots(figsize=(10, 10))
            Z = sc.fit_transform(dfwithoutname)

            Coord = pca.fit_transform(Z)    

            #L1 = list(pca.explained_variance_ratio_[0:6])
            #L1.append(sum(pca.explained_variance_ratio_[6:31]))

            plt.pie(pca.explained_variance_ratio_, labels=dfwithoutname.columns, autopct='%1.3f%%')
            
            
            plt.savefig(filename)
        #plt.show()
        return fig

    def plot_corrrelation_circle(self):
        filename = f'png/{self.file_name}plot_corrrelation_circle.png'
        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")

            img = mpimg.imread(filename)
            fig, axes = plt.subplots(figsize=(10, 10))
            plt.imshow(img)
            
        else :                        
            pca = PCA()
            dfwithoutname = self.get_numerical_var()        
            sc = StandardScaler()
            Z = sc.fit_transform(dfwithoutname)

            Coord = pca.fit_transform(Z)
            
            racine_valeurs_propres = np.sqrt(pca.explained_variance_)
            corvar = np.zeros((len(pca.explained_variance_ratio_),len(pca.explained_variance_ratio_)))
            for k in range(len(pca.explained_variance_ratio_)):
                corvar[:, k] = pca.components_[:, k] * racine_valeurs_propres[k]

            # Délimitation de la figure
            fig, axes = plt.subplots(figsize=(10, 10))
            axes.set_xlim(-1, 1)
            axes.set_ylim(-1, 1)

            # Affichage des variables
            for j in range(len(pca.explained_variance_ratio_)):
                plt.annotate(dfwithoutname.columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
                plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.6, alpha=0.5, head_width=0.03, color='b')

            # Ajout des axes
            plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
            plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

            # Cercle et légendes
            cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
            axes.add_artist(cercle)
            plt.xlabel('AXE 1')
            plt.ylabel('AXE 2')

            plt.savefig(filename)
        #plt.show()
        return fig

    def plot_groupement_from_key(self, key):
        filename = f'png/{self.file_name}pca_scatterplot.png'

        file_exists = os.path.exists(filename)
        if file_exists :
            print("-------------- file exists")

            img = mpimg.imread(filename)
            fig, axes = plt.subplots(figsize=(10, 10))
            plt.imshow(img)
            
        else :
            print("-------------- does not  exists")

            pca = PCA()
            dfwithoutname = self.get_numerical_var()        
            sc = StandardScaler()
            Z = sc.fit_transform(dfwithoutname)
            target = dfwithoutname[key]
            dfwithoutname.info()
            Coord = pca.fit_transform(Z)

            PCA_mat = pd.DataFrame({'AXE 1': Coord[:, 0], 'AXE 2': Coord[:, 1], 'target': target})

            PCA_mat.info()

            #(t) Représenter ces coordonnées colorées en fonction du diagnostic, à l'aide la fonction scatterplot() de seaborn*.
            fig, axes = plt.subplots(figsize=(10, 10))
            
            #plt.figure(figsize=(10, 10))

            sns.scatterplot(x='AXE 1', y='AXE 2', hue=target, data=PCA_mat)
            
            plt.savefig(filename)
            #
            # plt.show()
            return fig
        
