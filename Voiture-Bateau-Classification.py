#Importation des bibliotheques necessaires

import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle


#Les categories des images 
Categories=['car','ship']



flat_data_arr=[]
target_arr=[]

#Chemin de la BD
datadir='./DB'

#Parcourir les categories
for i in Categories:
  print(f'loading... category : {i}')
  path=os.path.join(datadir,i)
  #PreTraitement sur chaque image et les stocker dans le tableau "flat_data_arr"
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(150,150,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
  print(f'loaded category:{i} successfully')

#Affectation "image : etiquette categorie" dans le dataframe "df"
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target
df



#Les categories des images 
Categories=['car','ship']



flat_data_arr=[]
target_arr=[]

#Chemin de la BD
datadir='./DB'

#Parcourir les categories
for i in Categories:
  print(f'loading... category : {i}')
  path=os.path.join(datadir,i)
  #PreTraitement sur chaque image et les stocker dans le tableau "flat_data_arr"
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(150,150,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
  print(f'loaded category:{i} successfully')

#Affectation "image : etiquette categorie" dans le dataframe "df"
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target
df



#inputs
x=df.iloc[:,:-1]
#outputs
y=df.iloc[:,-1]

#division des donnees en 2 parties : donnees d'apprentissage et de test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')



#Apprentissage et construction de model

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
#svc : support vector classifier
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
#Parcourir les hyperparametres de "SVC" et selectionner les meilleurs parametres pour notre model avec "GridSearchCV"
model=GridSearchCV(svc,param_grid)
print("--")
model.fit(x_train,y_train)
print("The Model is trained well with the given images")
#Les meilleurs parametres
model.best_params_