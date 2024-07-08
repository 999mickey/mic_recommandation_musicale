import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score , matthews_corrcoef , confusion_matrix , average_precision_score
from sklearn.metrics import mean_squared_error , root_mean_squared_error

from sklearn.utils import resample

try:
    import StringIO 
except ImportError:
    from io import StringIO 
import sys
import os

class mic_model_manager():    
    """
    mic_model_manager gère des modèles de classification, régression et clustering applicable à la recommandation musicale
        
        Méthodes
        generate_data_and_test(...)

            La génération des données generate_data_and_test prend en argument : feature_list_name,target_name ,train_size,test_size,frac
                -feature_list_name : la liste des des variabless d'entrée des modèles
                -target_name : le nom de la variable cible
                -train_size : la taille ou proportion des donnée d 'entraitrement
                -test_size : la taille ou proportion des données de test
                -frac : optionnel, permet de réduire les ligne des données d 'entrée

                attention à  ne pas trop réduire les données avec frac, on peutt avoir plus de donnée d'entrainement et/ou de test qu'il ny en a

        rescale_data()
        log_repartion_var()

        set_resamplers(resamplers_param)
        resample_data(resampler_name):

        set_model_params(models_param)
        search_for_best_params()
        search_for_best_params_after_resampling()        
        log_best_params() 

        compute_target(namep=None, modelp= None)
        compute_targets()
        compute_targets_after_resampling()        

        collect_scores(self,name)
        get_accuracy(self,namep)
        get_recall(self,namep)        
        get_roc_au(self,namep)        
        get_rmse(self,namep)        
        get_mcc(self,namep)            
        get_cm(self,namep)        
        log_scores(self)

        catch_sdout()                
        uncatch_sdout()            
        print_logs_to_console()        
        write_logs_to_file(,name = None):                        
        log_df(self)           
    """
    
    def __init__(self,data_dir, data_file_name):
        self.data_file_name = data_file_name
        self.data_frame = pd.read_csv(data_dir + data_file_name)
        #self.data_frame = pd.read_csv(data_dir + data_file_name,index_col=0)
        self.log_df()
        self.models_params = dict()
        self.resamplers = dict()
        self.log_buf = StringIO()
        self.saved_log_buf = StringIO()
        self.X_train = any
        self.y_train = any
        self.X_test = any
        self.y_test = any
        self.y_pred = any
        self.best_paramslist = []
        self.best_params_dict = {}
        self.models_result = dict()
    ##########################
    #Générarion des données d 'entrées
    
    def generate_data_and_test(self,feature_list_name,target_name ,train_size,test_size,frac=None):
        print("************************* generate_data_and_test ******************************")   
        if frac != None:
            self.data_frame = self.data_frame.sample(frac=frac, random_state=42)
        self.X_train , self.X_test , self.y_train , self.y_test = train_test_split(
            self.data_frame[feature_list_name],
            self.data_frame[target_name] ,
            train_size=train_size,
            test_size=test_size,            
            random_state=666
            )
                
        self.rescale_data()
    def limit_df_to_oneuser(self,user_id):                
        self.data_frame = self.data_frame[self.data_frame['user_id'] == user_id]
        

    def rescale_data(self):
        print("************************* rescale_data ******************************")  

        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
    
    def log_repartion_var(self,column_name):
        print(self.data_frame[column_name].value_counts())
        df = self.data_frame
        percentP =  df[df[column_name] == 1].value_counts() .sum()*100/df[column_name].value_counts().sum()    
        percentN =  df[df[column_name] == 0].value_counts() .sum()*100/df[column_name].value_counts().sum()                    
        
        print("Poucentage de sentiment positif (1) : ",str(percentP))
        print("Poucentage de sentiment negatif (0) : ",str(percentN))
    ##########################
    #Resampling        
    def set_resamplers(self,resamplers_param):
        print("************************* set_resamplers ******************************")  

        self.resamplers = resamplers_param
        for res_name, res in self.resamplers.items():        
            print("set_resamplers model_name = ",res_name)        
            print(res)

    def get_resampler(self,resampler_name):
        print("************************* get_resampler ******************************")  

        for res_name, res in self.resamplers.items():        
            if res_name == resampler_name:
                return res
        return None                
        
    def resample_data(self,resampler_name):
        print("************************* resample_data ******************************")  
        resampler = self.get_resampler(resampler_name)            
        
        print("resample_data before len(Xtrain) = ",len(self.X_train))

        self.X_train, self.y_train = resampler.fit_resample(self.X_train, self.y_train)

        print("resample_data after len(Xtrain) = ",len(self.X_train))

        #exit()

        #self.X_test_res , self.y_test_res = resampler.fit_resample(self.X_test, self.y_test)    

    ##########################
    #Définition des modèle , et paramètres à pratir d 'un dicionnaire . Ex
    #model_params = {    
    #'LogisticRegression': {'model': LogisticRegression(solver='liblinear', random_state= 42),'params': { 'solver':['liblinear',],
    #        'C': [1, 5, 10],'penalty': ['l1', 'l2'] } },
    def set_model_params(self,models_param):
        print("************************* set_model_params ******************************")  
        self.models_param = models_param
        #for model_name, params in self.models_param.items():        
        #    print("set_model_params model_name = ",model_name)        

    #Recherche des meilleurs paramètres pour les modèles
    def search_for_best_params(self):
        print("************************* search_for_best_params ******************************")  
        self.best_paramslist = []
        for model_name, params in self.models_param.items():        
            print("model_name = ",model_name)
            clf = GridSearchCV(estimator=params['model'],param_grid=params['params'])            

            clf.fit(self.X_train, self.y_train)    

            print(f"Best model for [{model_name}]:")
            print(f"Best accuracy: {clf.best_score_}")
            print(f"Best parameters: {clf.best_params_}")                                        
            self.best_paramslist.append({model_name:clf.best_params_})
            self.best_params_dict[model_name] = {'best_params':clf.best_params_}
            params['model'].set_params(** clf.best_params_)        
            print("Score = ",clf.score(self.X_test, self.y_test))        
            #save best params in dictionary
            self.models_param[model_name]["best_param"] = clf.best_params_
            #print(self.models_param)
            
            print("="*20)

    #Recherche des meilleurs résultats après resmapling
    def search_for_best_params_after_resampling(self):       
        print("************************* search_for_best_params_after_resampling ******************************")  
        for res_name, res in self.resamplers.items():        
            self.resample_data(res_name)

            self.search_for_best_params()
            
    
    def log_best_params(self):      
        print("************************* log_best_params ******************************")  
        for name in self.models_param:
            print(name," best param are :",self.models_param[name]["best_param"])            

    ##########################
    #Calcul des prédictions
    def compute_targets(self,res_name = None):
        print("************************* compute_targets ******************************")  
                        
        for name , param in self.models_param.items():            
            
            model = param['model']            

            #définition des meilleurs modèles
            model.set_params(**self.best_params_dict[name]['best_params'])
            if res_name == None:
                self.compute_target(name , model)
            else :
                self.compute_target(name , model,res_name)
            

    def compute_target(self,namep, modelp, resample_name=''):  
            
            print("************************* compure_target ******************************")  
                        
            modelp.__init__()
            
            modelp.fit(self.X_train,self.y_train)

            self.y_pred = modelp.predict(self.X_test)

            score = modelp.score(self.X_test,self.y_test) 
            print("\nScore ( mean accuracy ) obtenu avec entrainement du modèle [",namep,"] sur les échantillons de test ",score)                
            self.collect_scores(namep,resample_name)

    #Calcul des prédictions après resampling
    def compute_targets_after_resampling(self):        
        print("************************* compute_targets_after_resampling ******************************")  

        for res_name, res in self.resamplers.items():        

            self.resample_data(res_name)

            self.compute_targets(res_name)            
            
    ##########################
    #Récuperer les résultats de prédiction
    def  collect_scores(self,name, res_name = ''):
        print("************************* collect_scores ******************************")  
            
        cm = confusion_matrix(self.y_test,self.y_pred)
        
        ras = roc_auc_score(self.y_test,self.y_pred)
        
        mcc = matthews_corrcoef(self.y_test,self.y_pred)
        
        pcs =  average_precision_score(self.y_test,self.y_pred)
        
        acc = accuracy_score(self.y_test,self.y_pred)
        
        f1 = f1_score(self.y_test,self.y_pred)

        #rms = mean_squared_error(self.y_test,self.y_pred, squared=False)
        rms = root_mean_squared_error(self.y_test,self.y_pred)
                    
        results = {
            'Matrice De confusion':cm,                
            'MCC Score': mcc,
            'ROC-AUC SCORE': ras,
            'Precision-recall Score':pcs,
            'Accuracy Score':acc,
            'F1 Score':f1,
            'RMSE':rms
        }
        #print("results => ",results)
        self.models_result[name+"_"+res_name] = results            

        print("self.models_result[",name+'_'+res_name,"] => ",self.models_result[name+"_"+res_name])
        

    def get_accuracy(self,namep):

        for name , vals in self.models_result.items():
            if name == namep: 
                return vals['Accuracy Score']
    
    def get_recall(self,namep):

        for name , vals in self.models_result.items():
            if name == namep: 
                return vals['Precision-recall Score']
    
    def get_roc_au(self,namep):

        for name , vals in self.models_result.items():
            if name == namep: 
                return vals['ROC-AUC SCORE']
    
    def get_rmse(self,namep):

        for name , vals in self.models_result.items():
            if name == namep: 
                return vals['RMSE']                
        
    def get_mcc(self,namep):

        for name , vals in self.models_result.items():
            if name == namep:
                return vals['MCC Score']
            
    def get_cm(self,namep):

        for name , vals in self.models_result.items():
            if name == namep:
                return vals['Matrice De confusion']            

    def get_f1(self,namep):

        for name , vals in self.models_result.items():
            if name == namep:
                return vals['F1 Score']            

    def log_scores(self):
        print("************************* log_scores ******************************")   
        #for name in self.models_result:
        #    print("Scores de prédcition pour ",name," => \n",self.models_result[name],"\n")            
        print("---------------------------------log_scores resultsize = ",len(self.models_result) )
        for name , _ in self.models_result.items():
            print(name,"MCC Score = ",self.get_mcc(name))
            print("RMSE = ",self.get_rmse(name))            
            print("Accuracy Score = ",self.get_accuracy(name))
            print("Precision-recall Score = ",self.get_recall(name))
            print("ROC-AU SCORE = ",self.get_roc_au(name))
            print("F1 Score = ",self.get_f1(name))
            print("Matrice de Confusion = \n",self.get_cm(name))
    
    ##########################
    #redirection de la sortie console vers sortie console ou fichier via self.log_buf = StringIO()
    def catch_sdout(self):        

        sys.stdout = self.log_buf                 

    def uncatch_sdout(self):    
        self.saved_log_buf = self.log_buf
        sys.stdout.close()    
        sys.stdout = sys.__stdout__    

    def print_logs_to_console(self):
        
        s = self.saved_log_buf.getvalue()
        
        print(s)        

    def write_logs_to_file(self,name = None):

        s = self.log_buf.getvalue() 
    
        if name == None:                   
            file_name = self.data_file_name +"_res.txt"
            if os.path.isfile(file_name):
                file_name += "temp.txt"
            sys.stdout = open(file_name, 'w')
        else :
            file_name = name
            if os.path.isfile(file_name):                
                file_name += "temp.txt"
                if os.path.isfile(file_name):
                    print("warning file name [",file_name,"] already exists")
            sys.stdout = open(file_name, 'w')    
        sys.stdout.write(s)
                        
    def log_df(self):        
        df = self.data_frame
        df.info()
        print(df.describe())
        print(df.head(1))
    
