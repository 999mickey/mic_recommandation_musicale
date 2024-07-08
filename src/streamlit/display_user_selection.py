import streamlit as st
from fire_state import create_store, form_update , set_state , get_state
import pandas as pd
slot = "home_page"


@st.cache_data
def load_data(content_path,name):    
    print("------------------------------------------------------------------<load_data(",name,") called")
    with st.spinner('lecture des données...'):
        df = pd.read_csv(content_path + name)
        
        st.success('Chargement effectué!')                
        #init()
        set_state(slot, ("file_name", name))
        return df

def display_user_selection(formname,filter):
    with st.form(formname):
        #st.slider("Nombre d utilisateur à présenter",min_value= 1,max_value= 10, value=5, step=1, key="num_user_to_present")                        
        #st.slider("Nombre d utilisateur à présenter",min_value= 1,max_value= 10, value=5, step=1, key=formname)                        
        #nuser = st.session_state["num_user_to_present"]
        #nuser = st.session_state[formname]
        #print("nombre d'utlisateur = ",st.session_state[formname])
        nuser = 100

        userIds =[]
        userIds = filter.get_best_voter_users(nuser)
        st.dataframe(userIds)        

        userIds = userIds[filter.user_key_name]
        userIds = userIds.to_list()
        userId = ''
        userIds.insert(0, userId)
        lltext = 'Choix de l utilisateur ('+str(len(userIds)-1)+')'
        #userId = st.selectbox('Choix de l utilisateur', userIds,key="user_id")
        userId = st.selectbox(lltext, userIds,key=formname)
        print("Utilisateur selectionné => ",userId)        
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))
        return userId

def display_nb_pres(val,maxp=None):
    maxval = 10
    if maxp != None:
        maxval = maxp
    with  st.form("Nombre 3"):                
        st.slider("Nombre de lignes à présenter",min_value= 1,max_value= maxval,value=val, step=1, key="nb")
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))
        return st.session_state["nb"]
    return maxval


def display_nb_pres_selection():
    with st.form("Nombre 2"):                
        st.slider("Nombre d utilisateur similaires",min_value= 1,max_value= 10,value=5, step=1, key="nb_similar_user")
        
        st.slider("Nombre de préférences", min_value= 1,max_value= 10,value=5, step=1, key="nb_pref_user")

        st.slider("Nombre de prédiction user", min_value= 1,max_value= 10,value=5, step=1, key="nb_pred_user")
        
        st.slider("Nombre de prédiction Item", min_value= 1,max_value= 10,value=5, step=1, key="nb_pred_item")
                
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))

        k_close_param = st.session_state["nb_similar_user"]
        npref = st.session_state["nb_pref_user"]
        npreditem = st.session_state["nb_pred_item"]
        npreduser = st.session_state["nb_pred_user"]
        return k_close_param ,npref , npreditem , npreduser

def display_predictors():
    from surprise import NormalPredictor
    from surprise import SVD
    from surprise.prediction_algorithms.knns import KNNBasic

    with st.form("predictor"):                
        predictors = ["","NormalPredictor()","SVD()","KNNBasic()"]
        lltext = 'Choix de predictor ('+str(len(predictors)-1)+')'
        predictor = st.selectbox(lltext, predictors,key="predictor")
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))
        ltxt = "Le predicteur choisi est :"+predictor
        st.text(ltxt)
        if predictor != '':
            return  eval(predictor) ,predictor
    return None , None

#def display_trainig_params(predicor_name):
#    if predicor_name == "SVD()":
def display_nan_replacement_options(filter):
    llext = "Il y a " +str(filter.get_nan_in_column(filter.target_key_name) ) +" Nans dans a colonnes"+filter.target_key_name
    st.text(llext)
    with st.form("nan_replacement"):                
        replacement_options = ["","Enlever les Nans",'Remplacer les Nans par zero','Remplacer les Nans par la moyenne'] 
        lltext = 'Choix de remplacement des valuers absentes (' + str(len(replacement_options)-1) +')'
        replacementoption = st.selectbox(lltext, replacement_options,key="replacement_nan_opt")
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))

        ltxt = "L'option choisie est :"+replacementoption
        st.text(ltxt)
        if replacementoption == 'Enlever les Nans':
            print(filter)
            filter.remove_nan(filter.target_key_name)
        if replacementoption == 'Remplacer les Nans par zero':
            filter.replace_nan_by_zero(filter.target_key_name)
        if replacementoption == 'Remplacer les Nans par la moyenne':
            filter.replace_nan_by_mean(filter.target_key_name)
        
        llext = "Il y a " +str(filter.get_nan_in_column(filter.target_key_name) ) +" Nans dans la colonne "+filter.target_key_name
        st.text(llext)
        




