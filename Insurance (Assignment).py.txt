# CALLING LIBRARY
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
###############################
# READING DATA
data=pd.read_csv('D:/ANIRUDH/DATA SCIENCE PROJECT/1.Insurance case study/insurance.csv')
###############################
data.columns # CHECKING COLUMNS
data.describe()
data.info()
data.isna().sum() # CHECKING FOR NULL VALUES
sns.heatmap(data.loc[:,('age','bmi','children','smoker','charges')].corr(),annot=True)
# SO SMOKER AND CHARGES HAVE 79% RELATION WITH EACH OTHER
###############################
# COMPARING AGE , BMI WITH CHARGES
# ABOVE MENTIONED COLUMNS ARE INTEGER, SO WE HAVE TO USE SCATTER FOR COMPARISON
plt.scatter(x=data['age'],y=data['charges']) # UNABLE TO FETCH INFORMATION
plt.scatter(x=data['bmi'],y=data['charges']) # UNABLE TO FETCH INFORMATION
# HERE DATA IS TOO DISCRETE SO WE ARE UNABLE TO FETCH INFORMATION

# FOR LABEL VS INTEGER , WE HAVE TO USE BOXPLOT
sns.boxplot(x=data['sex'],y=data['charges']) # MALES HAVE HIGH CHARGES
sns.boxplot(x=data['smoker'],y=data['charges']) # IF PERSON IS SMOKER THEN HE/SHE HAS TO PAY HIGH CHARGES
sns.boxplot(x=data['region'],y=data['charges']) # SOUTHEAST REGION HAVE HIGH CHARGES AS COMPARED TO OTHER REGIONS
sns.boxplot(x=data['children'],y=data['charges']) # FAMILY HAVING 2 CHILDREN HAVE MORE CHARGES
#################################
# CONVERTING LABELS INTO NUMERICAL FORM WITH THE HELP OF LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
columns=['sex','smoker','region']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
#################################
# SEPARATE DATA INTO INPUT AND OUTPUT 
x=data.drop(['charges'],axis=1)
y=data['charges']
#################################
# SEGREEGATE DATA INTO TRAINING AND TESTING
# WE WILL USE TRAIN_TEST MOUDLE FROM SKLEARN.MODEL_SELECTION
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#################################
# NOW,WE HAVE TO CALL REGRESSOR FOR OUR PREDICTIONS
# USE LINEAR REGRESSION FROM SKLEARN.LINEAR MODAL
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
# NOW ,ADD INPUT AND OUTPUT TO THE REGRESSOR FOR FUTURE PREDICTIONS
regressor.fit(x_train,y_train)
#################################
# TIME TO CHECK SLOPE AND INTERCEPT
regressor.coef_ 
regressor.intercept_ 
#################################
# TAKE Y_PRED FROM X_TEST
y_pred=regressor.predict(x_test)
#################################
# CHECK ACCURACY OF THE MODAL
# USE METRICS FROM SKLEARN TO CHECK ERROR
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) # 5663.358417062193
metrics.mean_absolute_error(y_test, y_pred) # 3998.2715408869726
metrics.r2_score(y_test, y_pred) # 0.7962732059725786
#PREDICTION IS GOOD AS ERROR IS LESS 
#  ------------------------------END----------------------------------








