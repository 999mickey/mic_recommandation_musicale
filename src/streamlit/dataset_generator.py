import pandas as pd
import numpy as np
import random

def generate_list_name(n,name):
    llist = []
    for i in range(0,n):
        llist.append(name+str(i))
    return llist

#nuser = 30
#ntrack = 200




notelist = [0,1,2,3,4,5,9,7,8,9,10]
nlist = len(notelist)

#nbvotesmax = 200


genre_list = ['pop','metal','classic','rap','jazz','disco','funk','rock'] #8 genres
#genre_list = ['pop','metal']
genre_list = ['pop','metal','classic','rap','jazz']


#nbvotesmax = 10
#nbvotesmin = 4
#nuser = 10
#ntrack = 20


df = pd.DataFrame( columns = ['user_id','user_likes', 'track_id', 'sentiment_score','genre'])
class Track():
    def __init__(self,num,genre):
        self.name = 'track'        
        self.name += str(num)        
        self.name += '_'
        self.name += genre
        self.genre = genre


class User():
    def __init__(self,numuser,nbvotesmin,nbvotesmax, ntrasks):
        self.name = 'usr'        
        self.taste = []
        self.name += str(numuser)
        self.ntracks = ntrasks
        self.votesmin = nbvotesmin
        self.votesmax = nbvotesmax
        self.nbvotes = random.randint(nbvotesmin,nbvotesmax)                
        #print("self.nbvotes = ",self.nbvotes)
        ##toberemoved
        #self.nbvotes = nbvotesmax                
        ##
        self.keeptrack_list = []
        self.track_list = []
        self.track_list = self.genrate_track_listwith_genres()
        print("len(self.track_list) = ",len(self.track_list) )
        self.set_taste()
        self.set_tracks()
        #self.log()
    def get_random_track(self,genre,number):
        
        locallist = self.get_tracks_ingenre(genre)    
        
        retlist = []
        for i in  range(0,number):
            if len(locallist)==0:
                return retlist
            randnum =  random.randint(0,len(locallist)-1)
            retlist.append(locallist[randnum])
            locallist.remove(locallist[randnum])
        return retlist

    def get_random_genres(self,listgenre,num):
        retList = []
        #print("get_random_genres(listgenre,num): ",listgenre)
        nlist  = listgenre.copy()
        for i in range(0,num):                
            randnum =  random.randint(0,len(nlist)-1)
            retList.append(nlist[randnum])
            nlist.remove(nlist[randnum])
            
        #print("------------>",genre_list)
        return retList

    def set_tracks(self):
        max = self.nbvotes
        #¢hoose random tracks in genre
        print("max = ",max)
        print("len(self.taste) = ",len(self.taste))
        #maxpergenre = int(max/len(self.taste))
        #maxpergenre = int(len(self.taste))
        maxpergenre = max
        print("maxpergenre = ",maxpergenre)

        for genre in self.taste:

            numberForgenre = random.randint(1,maxpergenre)
            print("numberForgenre = ",numberForgenre)
        
            ltracklist = self.get_random_track(genre,numberForgenre)
            #print(ltracklist)
            self.keeptrack_list += ltracklist 
            
            if max <= 0:
                return             
    def get_tracks_ingenre(self,genre):
        locallist = []
        for track in  self.track_list:
            if track.genre == genre:
                locallist.append(track)
        return locallist                    

    def genrate_track_listwith_genres(self):
        track_list = []
        for genre in genre_list:        
            for i in range (0,self.ntracks):
                track = Track(i,genre)
                track_list.append(track)
        return track_list                

    #track_list = genrate_track_listwith_genres()
            
    def set_taste(self):
        #genre_list
        self.number_of_genre = random.randint(1,3)
        self.number_of_genre = random.randint(1,len(genre_list))
        ##toberemoved
        #self.number_of_genre = 1                
        ##
        self.taste = self.get_random_genres(genre_list,self.number_of_genre)
        for taste in self.taste:
            self.name += "_"
            self.name += taste
        print(self.taste)
            
    def log(self):
        print("name = ",self.name)
        print("nombre de votes  = ",self.nbvotes)
        print("gouts = ",self.taste)
        print("nombres de tracks = ",len(self.keeptrack_list))
        print("track ")
        #for track in self.keeptrack_list:
        #    print(track.name) 

#user = User(0)            
"""def generate_user(num):
    user_list = []
    for i in range(0,num):                
        newuser = User(i)        
        user_list.append(newuser)
    return user_list 
user_list = generate_user(nuser)"""

def get_random_note(startinterval , stopintervale):
    tmplist=notelist[startinterval:stopintervale]
    #simule l absence de note
    tmplist.append(np.nan)
    num =  random.randint(0,len(tmplist)-1)
    return tmplist[num]

def generate_df_simu(nuserp , ntrackname, ngenre,ntrack,minvotesp,maxvotesp):
    ltext = "generate_df_simu("+str(nuserp)+ ","+str(ntrackname)+ ","+ str(ngenre) +","+str(ntrack)+"," + str(minvotesp) +"," +str(maxvotesp)+")"
    print(ltext)
    #
    genre_list = ['pop','metal','classic','rap','jazz','disco','funk','rock'] #8 genres
    genre_list = genre_list[0:ngenre]
    #
    df = pd.DataFrame( columns = ['user_id','user_likes', 'track_id', 'sentiment_score','genre'])
    #for user in user_list:
    for i in range(0,nuserp):
        print(i)
        #print("user ---------->",user)
        user = User(i,minvotesp,maxvotesp,ntrack)        
        #user.log()
        for track in user.keeptrack_list:
            note = get_random_note(0,10)            
            
            df.loc[len(df)] = [user.name,user.taste, track.name,note,track.genre]

    print("nombre d utilisateur = ",nuserp)
    print("nombre de tracks name = ",ntrack)

    print("nombre de genre = ",len(genre_list))
    print("nombre total de chansons  = ",len(user.track_list))
    print(ltext)
    print(df.shape)
    return df


#df = generate_df_simu(5 , 20, 3,30,4,10)

#print(df.head(20))
#print(df.tail(20))
#outfilename = "simulation_u"+str(nuser) + "-t"+ str(ntrack) + "_g" + str(len(genre_list)) + "_m" + str(nbvotesmax)+".csv"
#outfilename = "test.csv"

#print("save dataframe in "+outfilename)
#df.to_csv(outfilename)



