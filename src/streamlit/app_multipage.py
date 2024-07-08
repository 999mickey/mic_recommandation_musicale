import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")
st.write("### Introduction Application de Test de Recommantions Musicales")
if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

my_input = st.text_input("Input a text here", st.session_state["my_input"])
submit = st.button("Submit")
if submit:
    st.session_state["my_input"] = my_input
    st.write("You have entered: ", my_input)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import io
import time
import sys,os

sys.path.append(os.path.realpath('..'))


from models.mic_filtering_class import  mic_content_filtering ,mic_base_filter , mic_hybrid_filtering
from input_variables_setter import input_variables_setter
from features.mic_data_selection import mic_data_selection
from visualization.visualize import mic_vizualizer
from fire_state import create_store, form_update , set_state , get_state

content_path = '../../Data/'

slot = "home_page"

def get_session_state(key):
    if key in st.session_state:
        return st.session_state[key]
    return None


#load_count , file_name, song_name , artist_name , user_id ,my_data , my_visualizer , my_content_filter = create_store(slot, [        
create_store(slot, [        
    ("load_count", 0),        
    ("file_name", ''),        
    ("close_song_name",''),
    ("close_artist_name",''),
    ("user_id",''),
    ("mic_data", None)  ,  
    ("mic_viz",None)    ,
    ("mic_content_filtering",None)  ,
    ("content_path",content_path)
])
load_count = get_state(slot,"load_count")
file_name = get_state(slot,"file_name")
close_song_name = get_state(slot,"close_song_name")
close_artist_name = get_state(slot,"close_artist_name")

if close_artist_name == '' and get_session_state('close_artist_name'):
    close_artist_name = get_session_state('close_artist_name')
    set_state(slot, ("close_artist_name", close_artist_name))


user_id = get_state(slot,"user_id")
my_data = get_state(slot,"mic_data")
my_visualizer = get_state(slot,"mic_viz")
my_content_filter = get_state(slot,"mic_content_filtering")
                      

print("---------------------------------->IN \nst.session_state = ",st.session_state)

print("load_count => ", load_count)        
print("file_name => ", file_name)        
print("close_song_name => ",close_song_name)
print("close_artist_name => ",close_artist_name)

print("user_get_stateid => ",user_id)

print("mic_data => ", my_data)   

print("mic_viz",my_visualizer)    
print("mic_content_filtering",my_content_filter)        
print("---------------------------------->IN \n ")

load_count = get_state(slot,"load_count")

load_count += 1
set_state(slot ,("load_count",load_count))

current_filename = get_state(slot,'file_name')


#micdata = mic_data_selection()
if get_state(slot,'mic_data') == None :
    print("mic_base_filter allocation")
    my_data = mic_base_filter()

if get_state(slot,'mic_viz') == None :
    print("mic_vizualizer_filter allocation")
    my_visualizer = mic_vizualizer()

if get_state(slot,'mic_content_filtering') == None :
    print("mic_content_filtering_filter allocation")
    my_content_filter = mic_content_filtering()    

