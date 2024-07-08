import pandas as pd
import sys
import time
starttime = time.time()
inpath  = "../dataIn/"
inpath  = "dataIn/spotify/"
#inpath  = "../dataIn/kaggle/"
#inpath  = "../../../../data"
#inpath  = "../dataIn/test/"
import os
import sys

"""full_path = inpath + "twitter_all_merged.csv"
full_path = inpath +"module4_cleaned.csv"

df = pd.read_csv(full_path,on_bad_lines='skip')
#Attention a ne pas écraser les fichier csv
sys.stdout = open(full_path+".txt", 'w')
print(full_path)
print(df.info())
print(df.describe())
print(df.head(5))
sys.stdout.close()

exit()"""
directory_path = './'
#directory_path = inpath

def treatfile(name):
	df = pd.read_csv(name,on_bad_lines='skip')
	sys.stdout = open(name+".txt", 'w')
	print(name)
	print(df.info())
	print(df.describe())
	print(df.head(5))
	sys.stdout.close()


def list_files_recursive(path='.'):
	if os.path.isdir(path):
		for entry in os.listdir(path):
			full_path = os.path.join(path, entry)
			if os.path.isdir(full_path):
				list_files_recursive(full_path)
			else:
				
				if entry.endswith(".csv"):
					#
					treatfile(full_path)				
					
				else :
					print("bad file path Nothing to do, file shall be with .csv")	
	else:
		if path.endswith(".csv"):
			#
			treatfile(path)										
		else :
			print("bad file path Nothing to do, file shall be with .csv")	


print("input path = ",directory_path)

if len(sys.argv) > 1:
	directory_path = sys.argv[1]
	list_files_recursive(directory_path)
else:
	print("No parameters wher given...should be path file with .csv extension or directory ")
	print("Nothing tot do")






			    

# Specify the directory path you want to start from


sys.stdout = sys.__stdout__
stoptime = time.time()

print("Process took",stoptime - starttime," sec")

"""  sys.stdout = open(inpath + f+".txt", 'w')
            
            #logging.basicConfig(filename=inpath + f+".txt", level=logging.INFO)
            df = pd.read_csv(inpath + f)
            print(df.info())
            sys.stdout.close()
   """         