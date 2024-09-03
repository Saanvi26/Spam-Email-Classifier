import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('mail_data.csv')


data=df.where((pd.notnull(df)),'')

data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1

X=data['Message']
Y=data['Category']

# split the data as train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

# feature extraction 
feature_extraction = TfidfVectorizer(min_df =1, stop_words = 'english', lowercase=True)

# converting text data to feature vectors
X_train_feature = feature_extraction.fit_transform(X_train) #use fit_transform for training data
X_test_feature = feature_extraction.transform(X_test)    #use transform for testing data
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

# training the model
model=LogisticRegression()
model.fit(X_train_feature,Y_train)

# evaluation of the model
prediction_on_training_data=model.predict(X_train_feature)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print(accuracy_on_training_data)

# prediction on test data
prediction_on_test_data=model.predict(X_test_feature)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print(accuracy_on_test_data)


# testing the model with one sample mail :-
input_your_mail=["He will, you guys close?"]
input_data_feature=feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_feature)
# print(prediction)
if prediction[0]==0:
    print("Spam mail")
else:
    print("Ham mail")
    
