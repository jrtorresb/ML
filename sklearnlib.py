

import pandas as pd # Para carga de datos
import numpy as np # Algebra Lineal
import matplotlib.pyplot as plt # Graficas
import seaborn as sns # Estadistica

data = pd.read_csv(df, sep=";") # lees el archivo en csv


 # Librería Machine Learning sklearn
    
 from sklearn.model_selection import train_test_split # parte 75-25 
    
# Ejemplo usando wine.data, wine.target
from sklearn import datasets
wine = datasets.load_wine()

 X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state = 0)
    
 
    
    
#-------------------------------------MODELOS----------------------------------------------------------

# Modelo REGRESIÓN LOGÍSTICA    

from sklearn.linear_model import LogisticRegression
wine_logreg = LogisticRegression().fit(X_train, y_train)
    
print("Regresión logística")
print("Train set score: {:.3f}".format(wine_logreg.score(X_train, y_train)))
print("Test set score: {:.3f}\n".format(wine_logreg.score(X_test, y_test)))


# Modelo Reggresion Lineal    
from sklearn.linear_model import LinearRegression      
wine_lr=LinearRegression().fit(X_train,y_train)

# Modelo   KNeighborsClassifier  
from sklearn.neighbors import KNeighborsClassifier
wine_knn=KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

# Modelo Support Vector Machine
from sklearn.svm import LinearSVC
wine_linear_svm = LinearSVC().fit(X_train, y_train)

# Modelo Gaussiano
from sklearn.naive_bayes import GaussianNB
wine_bayes = GaussianNB().fit(X_train, y_train)

# Modelo Arboles de Decision
from sklearn.tree import DecisionTreeClassifier
wine_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    
# Modelo Bosque Aleatorio
from sklearn.ensemble import RandomForestClassifier
wine_forest = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)


  
    
# ------------------------------Métricas--------------------------------------   
    
  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report
    
    
y_pred = lr.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
    
 # Puedes poner las etiquetas
 print(classification_report(Y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]))
    
 print("Acurracy: {:.2f}". format(accuracy_score(y_test, y_pred)))
    # Los de abajo solo para clasificación binaria
 print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
 print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
 print("F1: {:.2f}".format(f1_score(y_test,y_pred)))
    
    
#------------------------------Curva ROC y AUC----------------------------------------- 
    
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score