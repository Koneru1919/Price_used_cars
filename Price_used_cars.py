#!/usr/bin/env python
# coding: utf-8

# # Predicting the Price of the used Cars¶

# ### Data Set Description:

# The given data set has 6019 attributes with 14 columns.Each column represents the information regarding the information of the car purchased.
# 
# 
# The data set includes different catgorical features such as Fuel_Type,Transmission,Owner_Type and  also numerical features such as the Year,Kilometers_Driven,Mileage, Engine,etc.Before building the model to predict Price,diferent steps such as data analysis,data visualization,data cleaning have been implemented.
# 
# The data set has been taken from kaggle(https://www.kaggle.com/avikasliwal/used-cars-price-prediction).

# ### Importing all the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Loading the data frame

# In[2]:


df=pd.read_csv("D:\\Misc\\Projects_ML\\Car-price prediction\\train-data.csv")


# In[3]:


df.describe()


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


#Checking the null values in the data
df.isnull().sum()


# In[7]:


df=df.drop(["New_Price"],axis=1)


# In[8]:


# So,there are null values in "Mileage","Engine","Power,Seats","New_price".Therefore eliminiating null values in these eatures

null_features=["Mileage","Engine","Power","Seats"]
for i in null_features:
    df=df[df[i].notna()]
df = df.reset_index(drop=True)    


# In[9]:


#Converting Mileage into one unit ('km/kg'
Mileage_converted= []
for i in df.Mileage:
    if str(i).endswith('km/kg'):
        i = i[:-6]
        i = float(i)*1.40
        Mileage_converted.append(float(i))
    elif str(i).endswith('kmpl'):
        i = i[:-6]
        Mileage_converted.append(float(i))


# In[10]:


df['Mileage']=Mileage_converted


# In[11]:


#Considering only non-Null values in "Power" feature
df[pd.to_numeric(df.Power, errors = 'coerce').isnull()]


# In[12]:


#Splitting the strings and neglecting the units.The units are eliminated from here on.
df['Engine'] = df['Engine'].str.split().str[0]
df['Power'] = df['Power'].str.split().str[0]
df=df[df.Power != 'null']


# In[13]:


#Converting Engine and Power float data types
df['Engine'] = df['Engine'].astype(float)
df['Power'] = df['Power'].astype(float)


# In[14]:


#Checking the current data types
df.dtypes


# In[15]:


df['Power'] = df['Power'].astype(float)


# In[16]:


df.isnull().sum()


# In[17]:


df.shape


# In[18]:


#Checking the unique values for each feature and storing the feature and its unique values in dictionary
Check_list=[]
feature_unique={'feature':[],'unique':[]}
for i in df:
        feature_unique['feature'].append(i)
        feature_unique['unique'].append(df[i].nunique())
        
for j in range(12):
    print("The {} feature has {} unique values".format(feature_unique['feature'][j],feature_unique['unique'][j]))


# In[19]:


df= df.drop(['Unnamed: 0'],axis = 1)


# ## Data Visualization

# In[20]:


plt.figure(figsize=(15,17))
sns.countplot(df['Location'])


# In[21]:


plt.figure(figsize=(40,40))
sns.catplot(x = 'Owner_Type', y = 'Kilometers_Driven',hue = 'Fuel_Type',kind = 'bar', data = df)
plt.title('kilometers vs The owner_type')


# In[22]:


sns.countplot(x = 'Transmission', hue ='Owner_Type', data = df )
plt.title('Counting transamission based on owner type')


# In[23]:


sns.catplot(x ='Fuel_Type' , y = 'Price',kind = 'bar',data = df)
plt.title("Price vs Fuel_type")


# In[24]:


sns.catplot(x ='Owner_Type' , y = 'Price',kind = 'bar',data = df)
plt.title("Price vs Fuel_type")


# In[25]:


sns.catplot(x ='Transmission' , y = 'Price',kind = 'bar',data = df)
plt.title("Price vs Fuel_type")


# In[26]:


corr = df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap='Reds')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[27]:


plt.scatter(y = 'Price', x = 'Year',data = df)


# In[28]:


#Creating a new feature for total number of years old
df['Present_Year'] = 2021
df['No_Years_Old'] = df['Present_Year'] - df['Year']


# In[29]:


#One hot encoding for fuel_type feature
Fuel_new= df[['Fuel_Type']]
Fuel_new= pd.get_dummies(Fuel_new,drop_first=True)
Fuel_new.head()


# In[30]:


#One hot encoding for Transmission
Transmission_new = df[['Transmission']]
Transmission_new = pd.get_dummies(Transmission_new,drop_first=True)
Transmission_new.head()


# In[31]:


#One hot encoding for Location
Location_new = df[['Location']]
Location_new = pd.get_dummies(Location_new,drop_first=True)
Location_new.head()


# In[32]:


#Dropping unnecessary features 
df= df.drop(['Year','Present_Year','Name'],axis = 1)


# In[33]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[34]:


#Label encoding for Owner_type
df['Owner_Type'] = le.fit_transform(df['Owner_Type'])


# In[35]:


#Concatinating all the  new built features into a new data frame
new_df=pd.concat([df,Location_new,Fuel_new,Transmission_new],axis=1)
new_df.head()


# In[36]:


#Dropping unnecessary features from the new data frame
new_df.drop(["Location","Fuel_Type","Transmission"],axis=1,inplace=True)
new_df.head()


# In[37]:


new_df.describe()


# In[38]:


##New data frame has 22 features
new_df.shape


# In[39]:


new_df.info()


# ### Preprocessing of the Data

# In[40]:


#Splitting of unlabaeled and unlabeled data
X = new_df.loc[:,['Kilometers_Driven', 'Owner_Type',
       'Mileage', 'Engine', 'Power', 'Seats', 
       'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore',
       'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur',
       'Location_Kochi', 'Location_Kolkata', 'Location_Mumbai',
       'Location_Pune', 'Fuel_Type_Diesel', 'Fuel_Type_LPG',
       'Fuel_Type_Petrol', 'Transmission_Manual','No_Years_Old']]
X.shape


# In[41]:


y = new_df.loc[:,['Price']]
y.head()


# In[42]:


#Splitting the data into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# ### Model Building

# ### Training the models using different algorithms and predicting on the test data.Different algorithms considered are
# 
# 1.Linear regression
# 
# 2.Linear regression(Lasso regularization- L1 regularisation)
# 
# 3.Linear regression(Ridge Regression - L2 regularisation )
# 
# 4.Decision Trees
# 
# 5.Random Forest Regressor
# 
# 6.K Neightbours Regressor

# In[43]:


#Importing models
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[44]:


y_pred_lr = lr.predict(X_test)


# In[45]:


#Score on Linear regression
lr.score(X_test,y_test)


# In[46]:


from sklearn.linear_model import  Lasso, Ridge


# In[47]:


#Model built with Linear regressionwith lasso regression
lasso = Lasso()
lasso.fit(X_train, y_train)


# In[48]:


#Score on Linear regression with lasso(L1 regularization)
lasso.score(X_test,y_test)


# In[49]:


y_pred_lasso= lasso.predict(X_test)


# In[50]:


#Model built with Linear regression with Ridge regression
ridge=Ridge()
ridge.fit(X_train, y_train)


# In[51]:


#Score on Linear regression with ridge(L2 regularization)
ridge.score(X_test,y_test)


# In[52]:


y_pred_ridge=ridge.predict(X_test)


# In[53]:


#Model built with DecisionTrees 
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)


# In[54]:


y_pred_dtr=ridge.predict(X_test)


# In[55]:


#Score on DecisionTrees
dtr.score(X_test,y_test)


# In[56]:


#Model built with Random Forest regressor 
from sklearn.ensemble import RandomForestRegressor

rfr =RandomForestRegressor()

rfr = RandomForestRegressor(n_estimators=200)

rfr.fit(X_train, y_train.values.ravel())

y_pred_rfr = rfr.predict(X_test)


# In[57]:


#Score on Random Forest Regressor
rfr.score(X_test,y_test)


# In[58]:


from sklearn.neighbors import KNeighborsRegressor


# In[59]:


#Model built by K Neightbours Regressor
knnModel = KNeighborsRegressor(n_neighbors=8)
knnModel.fit(X_train, y_train)


# In[60]:


knnModel.score(X_test,y_test)


# In[61]:


y_pred_knn=knnModel.predict(X_test)


# In[62]:


#Finally,comparing different results obtained on the algorithms considered


# In[63]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("""For LinearRegression,  RMSE is {:.2f} \t MSE is {:.2f} \t MAE is{:.2f} \t R2 is {:.2f} \t """.format(
            np.sqrt(mean_squared_error(y_test, y_pred_knn)),mean_squared_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)))

print("""DecisionTreeRegressor, RMSE is {:.2f} \t MSE is {:.2f} \t MAE is{:.2f} \t R2 is {:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),mean_squared_error(y_test, y_pred_dtr),
            mean_absolute_error(y_test, y_pred_dtr), r2_score(y_test, y_pred_dtr)))

print("""RandomForestRegressor, RMSE is {:.2f} \t MSE is {:.2f} \t MAE is{:.2f} \t R2 is {:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),mean_squared_error(y_test, y_pred_rfr),
            mean_absolute_error(y_test, y_pred_rfr), r2_score(y_test, y_pred_rfr)))


# ### Conclusion
# Random Forest Regressor is the best choice for this data set using the above algorithms.

# ### References
# https://www.kaggle.com/avikasliwal/used-cars-price-prediction
