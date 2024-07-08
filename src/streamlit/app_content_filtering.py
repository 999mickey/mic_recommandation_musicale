
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import io
import time
import sys,os
import json


#sys.path.append(os.path.realpath('..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..',''))




from models.mic_filtering_class import  mic_content_filtering ,mic_base_filter , mic_hybrid_filtering, mic_collaborativ_filtering
from input_variables_setter import input_variables_setter , list_files_recursive 
from features.mic_data_selection import mic_data_selection
from visualization.visualize import mic_vizualizer
from dataset_generator import generate_df_simu
from fire_state import create_store, form_update , set_state , get_state

from surprise import SVD
from surprise import NormalPredictor
from surprise.prediction_algorithms.knns import KNNBasic


###spéciifc display
from display_user_selection import display_user_selection , display_nb_pres_selection,display_predictors 
from display_user_selection import display_nb_pres,load_data , display_nan_replacement_options


import time
start = time.time()


st.set_option('deprecation.showPyplotGlobalUse', False)
content_path = '../../Data/'

main_slot = "home_page"

def get_session_state(key):
    if key in st.session_state:
        return st.session_state[key]
    return None

#load_count , file_name, song_name , artist_name , user_id ,my_data , my_visualizer , my_content_filter = create_store(main_slot, [        
create_store(main_slot, [        
    ("load_count", 0),        
    ("file_name", ''),        
    ("close_song_name",''),
    ("close_artist_name",''),
    ("true_song_name",''),
    ("true_artist_name",''),
    ("user_id",''),    
    ("mic_data", None)  ,  
    ("mic_viz",None)    ,
    ("mic_content_filtering",None),
    ("mic_collaborativ_filtering",None),
    ("mic_hybrid_filtering",None),
    ("num_user_to_present",5)   , 
    ("data-frame",pd.DataFrame())
])


def init():
    set_state(main_slot, ("file_name", ''))
    set_state(main_slot, ("close_song_name", ''))
    set_state(main_slot, ("close_artist_name",''))
    set_state(main_slot, ("true_song_name",''))
    set_state(main_slot, ("true_artist_name",''))
    set_state(main_slot, ("user_id", ''))
    set_state(main_slot, ("mic_data", None))
    set_state(main_slot, ("mic_viz", None))
    set_state(main_slot, ("mic_content_filtering", None))
    set_state(main_slot, ("mic_collaborativ_filtering", None))
    set_state(main_slot, ("mic_hybrid_filtering", None))
    set_state(main_slot, ("num_user_to_present", 5))
    set_state(main_slot, ("predictor", None))

load_count = get_state(main_slot,"load_count")
current_filename = get_state(main_slot,"file_name")

user_id = get_state(main_slot,"user_id")
my_data = get_state(main_slot,"mic_data")
my_visualizer = get_state(main_slot,"mic_viz")
my_content_filter = get_state(main_slot,"mic_content_filtering")
my_collaborativ_filtering = get_state(main_slot,"mic_collaborativ_filtering") 
my_hybrid_filtering = get_state(main_slot,"mic_hybrid_filtering")
def printvar():
    print('true_artist  = ',true_artist_name)
    print('true_song  = ',true_song_name)
    print('close_artist  = ',close_artist_name)
    print('close_song  = ',close_song_name)


print("-------------->load_count => ", load_count)        

print("INNNNNNNNNNNNNNNNNNNNNNNN--------------------------------file_name >> ", current_filename)        

load_count = get_state(main_slot,"load_count")

load_count += 1
set_state(main_slot ,("load_count",load_count))

#micdata = mic_data_selection()
if get_state(main_slot,'mic_data') == None :
    print("mic_base_filter allocation")
    my_data = mic_base_filter()
    set_state(main_slot, ("mic_data", my_data))

if get_state(main_slot,'mic_viz') == None :
    print("mic_vizualizer_filter allocation")
    my_visualizer = mic_vizualizer()
    set_state(main_slot, ("mic_viz", my_visualizer))

if get_state(main_slot,'mic_content_filtering') == None :
    print("mic_content_filtering_filter allocation")
    my_content_filter = mic_content_filtering()    
    set_state(main_slot, ("mic_content_filtering", my_content_filter))

if get_state(main_slot,'mic_collaborativ_filtering') == None :
    print("mic_collaborativ_filtering allocation")
    my_collaborativ_filtering = mic_collaborativ_filtering()    
    set_state(main_slot, ("mic_collaborativ_filtering", my_collaborativ_filtering))
    
if get_state(main_slot,'mic_hybrid_filtering') == None :
    print("mic_hybrid_filtering allocation")    
    my_hybrid_filtering = mic_hybrid_filtering()    
    set_state(main_slot, ("mic_hybrid_filtering", my_hybrid_filtering))



current_filename = get_state(main_slot,"file_name")

#toberemoved
_="""current_filename = "ratings_flrd.csv"
current_filename = "simulationcurrent.csv"
set_state(main_slot, ("file_name", current_filename))

df = load_data(content_path,current_filename)        
input_variables_setter(current_filename,my_data)
input_variables_setter(current_filename,my_visualizer)
input_variables_setter(current_filename,my_content_filter)
input_variables_setter(current_filename,my_collaborativ_filtering)
input_variables_setter(current_filename,my_hybrid_filtering)
my_collaborativ_filtering.data_frame = df
my_data.data_frame = df
my_visualizer.data_frame = df
my_content_filter.data_frame = df
my_hybrid_filtering.data_frame = df

#end"""

df = pd.DataFrame()

print("Projet de Recommandation Musicale avec ["+str(current_filename)+"]")

pages=["Intro","Choisir les données", "Visualisation des données", "Filtrage Contenu","Filtrage Memoire ","Filtrage Hybride "]
#st.title("Projet de Recommandation Musicale avec ["+str(current_filename)+"]",anchor = 'title')
st.sidebar.title("Fichier selectionné : "+current_filename)

page=st.sidebar.radio("Aller vers", pages)

print("---page",page)

if page == pages[0] : 
    st.write("### Introduction Application de  test de Recommantions Musicales")
    st.write("Le Filtrage Contenu nécessite les valeurs intrinsèques des morceaux de musiques \
    telles que 'liveness', 'speechiness', 'danceability', 'valence', 'loudness', 'tempo', 'acousticness','energy', 'mode', 'key', 'instrumentalness'  ")
    st.write("Le Filtrage Mémoire et Hybride nécessite des identifiants d'utilisateurs et de chasons ainsi que des scores attribués aux morceaux ")
        
if page == pages[1] : 
    input_type , usernb_input , tracknb_input = create_store("input", [        
        ("input_type",''),   
        ("usernb_input",5),   
        ("tracknb_input",20),  
        

    ])
    input_type = get_state('input','input_type')
    st.write("### Choisir des données")
    print("### Choisir des données")

    with  st.form("input_type"):                
        choiceslist = ["","A partir d un fichier","En simulation"]
        lltext = 'Type d entrée ('+str(len(choiceslist)-1) + ' choix)'

        input_type = st.selectbox(lltext, choiceslist,key="input_type")
        st.form_submit_button(label="Valider ", on_click=form_update, args=(main_slot,))        

    print("input_type = ",input_type)
    if  input_type == 'En simulation':
            print("--------------> intype is generated")
            with  st.form("simu_params"):                
                usernb_input = st.number_input("Choisir un nombre d utilisateur =",value=5,min_value=3,max_value=20, placeholder="usernb_input")
                tracknamenb_input = st.number_input("Choisir le  nombre noms de morceaux =",value=20,min_value=10,max_value=200, placeholder="tracknamenb_input")
                genrenb_input = st.number_input("Choisir nombre de genres =",value=3,min_value=1,max_value=8, placeholder="genrenb_input")
                tracknb_input = st.number_input("Choisir nombre de morceaux =",value=40,min_value=10,max_value=200, placeholder="tracknb_input")
                minvote_nb = 4
                minvote_nb = st.number_input("Choisir nombre de votes min =",value=4,min_value=2,max_value=20, placeholder="minvote_nb")            
                maxvote_nb = 10
                #todo check for coherence with min votes num
                maxvote_nb = st.number_input("Choisir nombre de votes min =",value=10,min_value=10,max_value=200, placeholder="maxvote_nb")
                submit = st.form_submit_button(label="Valider ", on_click=form_update, args=(main_slot,))        
                #submit = st.button("Valider la simulation")
                if submit :
                    ltext = "generate_df_simu("+str(usernb_input)+ ","+str(tracknamenb_input)+ ","+ str(genrenb_input) +","+str(tracknb_input)+"," + str(minvote_nb) +"," +str(maxvote_nb)+")"
                    st.write( ltext)
                    df = generate_df_simu(usernb_input,tracknamenb_input,genrenb_input,tracknb_input,minvote_nb,maxvote_nb)
                    set_state(main_slot, ("data-frame",df))
                    input_variables_setter("simu",my_data)
                    input_variables_setter("simu",my_visualizer)
                    input_variables_setter("simu",my_content_filter)
                    input_variables_setter("simu",my_collaborativ_filtering)
                    input_variables_setter("simu",my_hybrid_filtering)
                    
                    my_data.data_frame = df       
                    my_content_filter.data_frame = df

                    if len(my_content_filter.features ):
                        my_content_filter.normalize_data()
                    
                    my_visualizer.data_frame = df
                    my_collaborativ_filtering.data_frame = df
                    my_hybrid_filtering.data_frame = df        
                    current_filename = "simu"
                    name ="simu"
                    set_state(main_slot, ("file_name", name))
                    st.write('Le fichier choisi est :', name)  
                    st.write("Forme :",df.shape)
                    buffer = io.StringIO()
                    dfinfo = pd.DataFrame(df.info())
                    df.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text("Info()")
                    st.text(s)
                    #st.text(dfinfo.head())
                    st.text("Head()")
                    st.dataframe(df.head())
                    st.text("Describe()")
                    st.dataframe(df.describe())        
            
    elif input_type == 'A partir d un fichier':
        print("--------------> intype is from file")
        datas=["","merge.csv","data.csv","simulation.csv","simulationcurrent.csv"]
        datas = list_files_recursive(content_path)
        datas.insert(0,'')
        lltext = 'Choix des données (' +str(len(datas)-1)+ ' choix)'
        
        name = st.selectbox(lltext, datas,key="data_name")
            
        #if current_filename != name and name != '': 
        if  name != '': 
    
            df = load_data(content_path,name)        
            set_state(main_slot, ("data-frame",df))
            st.title("Projet de Recommandation Musicale avec ["+str(name)+"]",anchor = 'title')
            st.sidebar.title("Fichier selectionné : "+name)
            set_state(main_slot, ("file_name", name))

            input_variables_setter(name,my_data)
            input_variables_setter(name,my_visualizer)
            input_variables_setter(name,my_content_filter)
            input_variables_setter(name,my_collaborativ_filtering)
            input_variables_setter(name,my_hybrid_filtering)

            my_data.data_frame = df       
            my_content_filter.data_frame = df

            if len(my_content_filter.features ):
                my_content_filter.normalize_data()
            
            my_visualizer.data_frame = df
            my_collaborativ_filtering.data_frame = df
            my_hybrid_filtering.data_frame = df        

            current_filename = name
            st.write('Le fichier choisi est :', name)  
            st.write("Forme :",df.shape)
            buffer = io.StringIO()
            dfinfo = pd.DataFrame(df.info())
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text("Info()")
            st.text(s)
            #st.text(dfinfo.head())
            st.text("Head()")
            st.dataframe(df.head())
            st.text("Describe()")
            st.dataframe(df.describe())        
        
        else :
            df = my_data.data_frame    
        

if page == pages[2] :
    st.write("### Visualisation des données : ",str(current_filename))
    df = get_state(main_slot,'data-frame')
    print(df.head())
    if len(df) != 0:
        st.write("Forme :",df.shape)
        
        st.text("Colmuns")
        st.dataframe(df.head(0))
        
        print("my_visualizer.is_content_to_be_filtered  = ",my_visualizer.is_content_to_be_filtered )
        if my_visualizer.is_content_to_be_filtered == True :                            
            ltext = "Présenter la Heatmap"
            if st.checkbox(ltext,key="check_heatmap") :
                with st.spinner('Calcul en cours...'):
                    fig = my_visualizer.plot_heatmap()
                    st.pyplot(fig)
            

            if my_visualizer.target_key_name != '' :
                checktitle = "Présenter le Distribution de "+my_visualizer.target_key_name
                if st.checkbox(checktitle,key="dist_target") :
                    with st.spinner('Calcul en cours...'):
                        plt.title("Distribution de "+my_visualizer.target_key_name)
                        st.pyplot(my_visualizer.plot_repartion_count_target(my_visualizer.target_key_name))

                title = "Présenter le cercle de corrélation "

                if st.checkbox("Cercle de corrélation",key="corr_target") :
                    with st.spinner('Calcul en cours...'):
                        plt.title("Cercle de corrélation ")
                        st.pyplot(my_visualizer.plot_corrrelation_circle())

                title = "Présenter l'analyse des principaux composants par "+ my_visualizer.target_key_name            
                if st.checkbox(title,key="pca") :
                    with st.spinner('Calcul en cours...'):
                        plt.title(title)
                        st.pyplot(my_visualizer.plot_groupement_from_key(my_visualizer.target_key_name))            

                title = "Présenter l'analyse des principaux composants en camenbert"            
                if st.checkbox(title,key="pc_cam") :
                    with st.spinner('Calcul en cours...'):
                        plt.title(title)
                        st.pyplot(my_visualizer.plot_pie_of_principal_comp())            

                

                

        if 'genre' in my_visualizer.data_frame.columns:
            title = "Présenter la répartition par 'genre"
            if st.checkbox(title,key="rep_target") :
                with st.spinner('Calcul en cours...'):
                    plt.title(title)
                    fig = my_visualizer.plot_repartion_count_target('genre')
                    st.pyplot(fig)
        #if my_visualizer.is_content_to_be_sparsed == True :  
        if my_visualizer.target_key_name != '' :
            ltext = "Présenter la répartition des scores [" + my_visualizer.target_key_name + "]"
            if st.checkbox(ltext,key="viz_plot_rep") :
                with st.spinner('Calcul en cours...'):
                    fig = my_visualizer.plot_repartion_count_target(my_visualizer.target_key_name,5)
                    st.pyplot(fig)

            ltext = "Présenter les morceaux les plus populaires [" + my_visualizer.item_key_name + "] / ["+my_visualizer.target_key_name+"]"
            if st.checkbox(ltext,key="_viz_most_pop") :
                with st.spinner('Calcul en cours...'):
                    fig = my_visualizer.plot_most_popular_tracks(10)
                    st.pyplot(fig)

            ltext = "Présenter les morceaux les mieux notés [" + my_visualizer.item_key_name + "] / ["+my_visualizer.target_key_name+"]"
            if st.checkbox(ltext,key="viz_best_rat") :
                with st.spinner('Calcul en cours...'):
                    fig = my_visualizer.plot_best_noted_tracks(10)
                    st.pyplot(fig)

            ltext = "Présenter les morceaux les mieux notés en moyenne[" + my_visualizer.item_key_name + "] / ["+my_visualizer.target_key_name+"]"
            if st.checkbox(ltext,key="viz_best_mean_rat") :
                with st.spinner('Calcul en cours...'):
                    fig = my_visualizer.plot_best_mean_noted_tracks(10)
                    st.pyplot(fig)
        
        
    else:
        print("no filename selectionned")

if page == pages[3]:
    create_store("content_filtering", [        
        ("close_song_name",''),
        ("close_artist_name",''),
        ("true_song_name",''),
        ("true_artist_name",''),
        
    ])
    df = get_state(main_slot,'data-frame')
    if len(my_content_filter.features) == 0:
        st.write("### Filtrage contenu avec ",str(current_filename), "impossible (valeurs musicales intrinsèques absentes )" )
    #elif current_filename != '':        
    elif len(df) != 0:
        st.write("### Filtrage contenu avec ",str(current_filename))
        if st.checkbox("Nettoyer les noms",key="clean_name") :
            true_artist_name = ''
            true_song_name = ''
            close_artist_name = ''
            close_song_name = ''
            set_state(main_slot, ("close_song_name", ''))
            set_state(main_slot, ("close_artist_name",''))
            set_state(main_slot, ("true_song_name",''))
            set_state(main_slot, ("true_artist_name",''))
            #st.session_state['clean_name'] = False

        else:    
            true_artist_name = get_state(main_slot,'true_artist_name')
            true_song_name = get_state(main_slot,'true_song_name')
            close_artist_name = get_state(main_slot,'close_artist_name')
            close_song_name = get_state(main_slot,'close_song_name')
      
            #choix approximamtif
            if close_artist_name == '' and close_song_name == '':    
                with  st.form("close artist and song"):                
                    close_artist_name = st.text_area('choisir le nom d atiste approximativement',key='close_artist_name')    
                    #close_song_name = st.text_area('choisir le nom de chanson approximativement',key='close_song_name')    
                    st.form_submit_button(label="Valider", on_click=form_update, args=(main_slot,))        
            #choix de l'artiste
            if close_artist_name != '':
                with  st.form("artist "):                
                    artist_list = my_content_filter.get_artist_closedto(close_artist_name)                
                    artist_list.insert(0,'')
                    lltext = 'Choix de l artiste (' +str(len(artist_list)-1)+ ' choix)'
                    true_artist_name = st.selectbox(lltext, artist_list,key="artist_name")
                    st.form_submit_button(label="Valider artist", on_click=form_update, args=(main_slot,))        
            #choix du morceau
            if true_artist_name != '':
                with  st.form("song "):                                               
                    #song_list = my_content_filter.get_artist_song_closedto(true_artist_name,close_song_name)
                    song_list = my_content_filter.get_all_artist_songs(true_artist_name)
                    song_list.insert(0,'')                
                    lltext = 'Choix de l artiste (' +str(len(song_list)-1)+ ' choix)'
                    true_song_name = st.selectbox(lltext, song_list,key="song_name")
                    st.form_submit_button(label="Valider song", on_click=form_update, args=(main_slot,))        
            #caclul des mtoceaux les plus proches
            if true_artist_name != '' and true_song_name != '':         

                ltext = 'Choix du nom d atiste : '+ true_artist_name
                st.text(str(ltext))    
                ltext = 'Choix du nom de chanson : ' + true_song_name
                st.text(str(ltext))    
                #print("my_content_filter.artist_key_name =",my_content_filter.artist_key_name)
                #print(my_content_filter.data_frame[my_content_filter.artist_key_name].head(10))
                num = 10
                num = display_nb_pres(20,100)
                                        
                songnum = my_content_filter.get_track_num_of_artist(true_artist_name,true_song_name)                
                
                print('--------------- songnum = ',songnum)
                with st.spinner('Calcul de morceaux les plus proches...'):
                    SongId , SongName , ArtistName , Distance = my_content_filter.content_filter_music_recommender(songnum, num)
                st.success('compute tracks Done!')                

                outputdf = pd.DataFrame()
                outputdf["song_id"] = SongId
                outputdf["artists"] = ArtistName
                outputdf["song_name"] = SongName
                outputdf["distance"] = Distance
                print(outputdf.head(num))
                st.write('Chanson le splus proches :')                    
                st.dataframe(outputdf.head(num))    
                            
    else :
        print('no filename selected')

if page == pages[4] : 
    create_store("memory_filtering", [        
    ("user_id", ''),      
    ("user_id2", '')      
    ])
    my_collaborativ_filtering.print_vars()

    df = get_state(main_slot,'data-frame')
    print("toberemoved my_collaborativ_filtering.user_key_name =", my_collaborativ_filtering.user_key_name)
    
    if my_collaborativ_filtering.user_key_name     == '':
        st.write("### Filtrage mémoire avec ",str(current_filename), "impossible (valeurs utilisateur absentes )" )
    #else :
    elif len(df) != 0:
        st.write("### Filtrage mémoire ",str(current_filename))

        my_collaborativ_filtering.print_vars()
        
        print(my_collaborativ_filtering.data_frame.info()    )
        userId =  display_user_selection("filt_mem_user",my_collaborativ_filtering)

        print("Utilisateur selectionné => ",userId)        
        if userId != '' :
            #k_close_param , npref , npreditem , npreduser = display_nb_pres_selection()
            st.text("Profondeur de présentation "+str(userId))
            nb = display_nb_pres(10,20)
            npref = 10
            npreduser = nb
            npreditem  = nb
            score_seuil = 0
            k_close_param = nb

            #user description
            st.text("Utilsateur "+str(userId))
            #print(my_collaborativ_filtering.get_user_description(userId))
            st.text(my_collaborativ_filtering.get_user_description(userId))

            print(my_collaborativ_filtering.get_user_description(userId))

            st.text("Préférences de "+str(userId))
            user_preferences = my_collaborativ_filtering.get_preferences(userId,score_seuil,npref)
            top = user_preferences.sort_values(my_collaborativ_filtering.target_key_name, ascending=False)  
            st.dataframe(top)    
            
            score_seuil = 0
            my_collaborativ_filtering.generate_notation_mattrix()
            simlilar_users = my_collaborativ_filtering.get_similar_users(k_close_param,userId)
            ret = pd.DataFrame({'Identifiant Utilisateur':simlilar_users.index, 'Similarité':simlilar_users.values})
            st.text("Utilisateurs similaires ")
            st.dataframe(ret)    
            
            st.text("Prédiction utillisateur "+str(userId))
            reco_user = my_collaborativ_filtering.pred_user(k_close_param,userId,npreduser)
            ret = pd.DataFrame({'Titre':reco_user.index, 'Score':reco_user.values})
            #ret = pd.DataFrame(reco_user)

            #st.dataframe(reco_user)    
            st.dataframe(ret)    

            st.text("Prédiction morceau "+str(userId))
            reco_item = my_collaborativ_filtering.pred_item( k_close_param,userId,npreditem).sort_values(ascending=False).head(npreditem)
            ret = pd.DataFrame({'Titre':reco_item.index, 'Score':reco_item.values})
            st.dataframe(ret)    
            set_state("memory_filtering", ("user_id", userId))
            
            userId2 = get_state("memory_filtering","user_id2")
                
            if userId2 == '':
                ltext =  'Choisir un autre utisateur que ['+userId+'] pour en calculer le coefficient de similarité'
                st.text(ltext)
                userId2 =  display_user_selection("moukfilt_mem_user2",my_collaborativ_filtering)
                

                print("userId2 = ",userId2)
                if userId2 != '':
                    simscore = my_collaborativ_filtering.get_users_similarity(userId,userId2)
                    letext = "Coefficient de similarité entre \n["+str(userId)+"] et \n["+str(userId2)+"] \n = \n" +str(simscore)
                    st.text(letext)
                    

if page == pages[5] : 
    create_store("hybrid_filtering", [        
    ("user_id", ''),      
    ("predictor", None),
    ("ispredictor_trained",False),    
    ])
    df = get_state(main_slot,'data-frame')
    if my_content_filter.user_key_name     == '':
        st.write("### Filtrage hybride avec ",str(current_filename), "impossible (valeurs utilisateur absentes )" )
    #else :
    elif len(df) != 0:
        st.write("### Filtrage hybride ",str(current_filename))
        #my_hybrid_filtering
        my_hybrid_filtering.generate_dateset_autofold()
        my_hybrid_filtering.data_frame.info()    
        my_hybrid_filtering.clean_columns()
        ispredictor_trained = False
        display_nan_replacement_options(my_hybrid_filtering)

        predictor , predictor_name = display_predictors()
        print("type(predictor) = ",type(predictor))
        set_state("hybrid_filtering", ("predictor", predictor))
        
        #if predictor_name == 'SVD()':        
        if predictor != None :
            print("predictor = ",predictor_name)
            #ltext = "Predicteur choisi :"+predictor_name
            #st.text(ltext)
            my_hybrid_filtering.generate_dateset_autofold()

            lltext = "Evaluation du modèle "+ predictor_name + ": " 
            st.text(lltext)
            with st.spinner('Calcul en cours...'):
                my_hybrid_filtering.evaluate_predictor(predictor)

            lltext = "Evaluation Erreur Absolue Moyenne (MAE): "+str(my_hybrid_filtering.average_eveluation_mae)
            
            st.text(lltext)
            lltext = "Evaluation Erreur Quadratique Moyenne (RMSE): "+str(my_hybrid_filtering.average_eveluation_rmse)
            st.text(lltext)                    

            ispredictor_trained = get_state("hybrid_filtering",'ispredictor_trained')
            if ispredictor_trained == False:
                if predictor_name == 'SVD()':
                    
                    #param_grid = {'n_factors': [100,150],
                    #    'n_epochs': [20,25,30],
                    #    'lr_all':[0.005,0.01,0.1],
                    #    'reg_all':[0.02,0.05,0.1]}
                    
                    param_grid = {'n_factors': [10,15],
                        'n_epochs': [20,25],
                        'lr_all':[0.005,0.1],
                        'reg_all':[0.02,0.1]}
                    #param_grid = {}

                    lltext = 'Paramètres d entrainememt :' + json.dumps(param_grid)
                    st.text(lltext)
                    if st.button("Rechercher les meilleurs paramètres"):    
                        with st.spinner('Calcul en cours...'):
                            ret = my_hybrid_filtering.predictor_ajustement(SVD,param_grid)
                            st.text("Les meilleurs paramères sont: ")                                            
                            st.text(json.dumps(my_hybrid_filtering.best_params_predictor))

                            st.text("Les meilleurs scores sont: ")                                            
                            st.text(json.dumps(my_hybrid_filtering.best_score))                                
                            #predictor = SVD(**my_hybrid_filtering.best_params_predictor)
                            predictor = SVD(n_factors= 100, n_epochs= 20, lr_all=0.005, reg_all= 0.1)
                            
                        set_state("hybrid_filtering", ("predictor", predictor))
                        set_state("hybrid_filtering", ("ispredictor_trained", True))                    

                if predictor_name == 'NormalPredictor()':
                    predictor = NormalPredictor()
                    param_grid = {}
                    lltext = 'Rechercher les meilleurs paramètres' + json.dumps(param_grid)
                    st.text(lltext)
                    if st.button("Entrainer le modèle"):    

                        with st.spinner('Calcul en cours...'):                        
                            ret = my_hybrid_filtering.predictor_ajustement(NormalPredictor,param_grid)
                            st.text("Les meilleurs paramères sont: ")                                            
                            st.text(json.dumps(ret))
                        
                        set_state("hybrid_filtering", ("predictor", predictor))
                        set_state("hybrid_filtering", ("ispredictor_trained", True))
                if predictor_name == 'KNNBasic()':
                    predictor = KNNBasic()
                    #n_neighbors=7,metric='minkowski'
                    #'manhattan'
                    #'chebyshev'
                    param_grid = {'name': 'cosine',
                        'user_based': 'False'
                    }
                    
                    #param_grid = {}

                    _="""lltext = 'Paramètres d entrainememt :' + json.dumps(param_grid)
                    st.text(lltext)
                    if st.button("Rechercher les meilleurs paramètres "):    
                        with st.spinner('Calcul en cours...'):
                            #ret = my_hybrid_filtering.predictor_ajustement(predictor,param_grid)
                            ret = my_hybrid_filtering.predictor_ajustement(KNNBasic,param_grid)
                            
                            st.text("Les meilleurs paramères sont: ")                                            
                            st.text(json.dumps(my_hybrid_filtering.best_params_predictor))

                            st.text("Les meilleurs scores sont: ")                                            
                            st.text(json.dumps(my_hybrid_filtering.best_score))                                
                            #predictor = SVD(**my_hybrid_filtering.best_params_predictor)
                            #predictor = SVD(n_factors= 100, n_epochs= 20, lr_all=0.005, reg_all= 0.1)
                            
                        set_state("hybrid_filtering", ("predictor", predictor))
                        set_state("hybrid_filtering", ("ispredictor_trained", True))                    

                    """
        
        if get_state("hybrid_filtering",'predictor') != None :        
            #default user
            #if st.checkbox("Utilisateur pa rdefaut",key="default_user") :
                            #chosen user
            nb = display_nb_pres(10,20)

            userId = my_hybrid_filtering.default_user
            st.text("Choisir un utilisateur pour calculer une prédiction ")            
            userId =  display_user_selection("file_hyb_user",my_hybrid_filtering)
            if userId != '':
                #k_close_param , npref , npreditem , npreduser = display_nb_pres_selection()
                npreditem = nb

                
                score_seuil = 0

                with st.spinner('Calcul en cours...'):
                    npreditem = nb
                    
                    st.text("Préférences de l'ustilisateur "+str(userId))            
                    user_preferences = my_collaborativ_filtering.get_preferences(userId,score_seuil,npreditem)
                    top = user_preferences.sort_values(my_collaborativ_filtering.target_key_name, ascending=False)
                    st.dataframe(top)    
                    predictor = get_state("hybrid_filtering",'predictor') 
                    
                    pred = my_hybrid_filtering.predict(userId,predictor,npreditem) 
                    st.text("Prédiction Pour l'utilisateur  "+str(userId))
                    st.dataframe(pred)    
                    


    


