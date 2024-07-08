import pandas as pd


#Titre de la chanson,Genre,User1,User2,User3,User4,User5,User6,User7,User8,User9,User10
#df1 = df1 .drop("Genre",axis=1)


def transformdf(df):
    melt= pd.melt(
        df,
        id_vars=["Titre de la chanson","Genre"],
        value_vars=["User1", "User2","User3","User3","User4","User5"
                    ,"User6","User7","User8","User9","User10"],
        var_name="User",
        value_name="Score",
    )
    return melt
df1 = pd.read_csv('funk.csv')
melt = transformdf(df1)
print(melt.shape)
##
df2 = pd.read_csv('gypsy_jazz.csv')
ret = transformdf(df2)
melt = pd.concat([melt,ret])
print(melt.shape)


df = pd.read_csv('pop.csv')
ret = transformdf(df)
melt = pd.concat([melt,ret])
print(melt.shape)


df = pd.read_csv('jazz.csv')
ret = transformdf(df)
melt = pd.concat([melt,ret])
print(melt.shape)

df = pd.read_csv('rock.csv')
ret = transformdf(df)
melt = pd.concat([melt,ret])
print(melt.shape)


df = pd.read_csv('soul.csv')
ret = transformdf(df)
melt = pd.concat([melt,ret])
print(melt.shape)


df = pd.read_csv('techno.csv')
ret = transformdf(df)
melt = pd.concat([melt,ret])
print(melt.shape)


#melt['track_id'] = melt['Genre'] + ' : ' + melt['Titre de la chanson']
melt['track_id'] = melt['Titre de la chanson'] + ' : ' +  melt['Genre'] 
ldict = {'Titre de la chanson':'Titre','User':'user_id','Genre':'genre','Score':'sentiment_score'}
melt = melt.rename(ldict,axis=1)
melt = melt.drop('Titre',axis=1)
melt["user_id"] = melt["user_id"].str.lower()
melt.style.set_properties(**{'text-align': 'left'})
melt.info()

print(melt.head())
#print()

melt.to_csv('merge.csv')

