#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# # ----LOAD THE DATA----

# In[2]:


path = r"C:\Users\Sachin Kumar\Downloads\Bengaluru_House_Data (1).csv"
df_orgl = pd.read_csv(path)


# In[3]:


df_orgl.shape


# In[4]:


df_orgl.head()


# # --Analysing The Data--

# In[5]:


df_orgl.info()


# In[6]:


df_orgl.describe()


# In[7]:


df = df_orgl.copy()


# In[8]:


def value_count(df):
    for var in df.columns:
        print(df[var].value_counts())
        print("-----------------------------")
    


# In[9]:


value_count(df)


# # Checking The Correlation Using Heatmap

# In[10]:


num_vars = ["bath","balcony","price"]
sns.heatmap(df[num_vars].corr(),cmap ='coolwarm',annot =True)


# Correlation of bath is greater than balcony with price

# # Preparing the Data for Machine Learning Model

# # 1. Data Cleaning
# 

# In[11]:


miss_data =df.isnull().sum()
miss_data


# In[12]:


#Checking the percentage of missing data
df.isnull().sum()/df.shape[0]*100


# In[13]:


#Visualizing the data of missing values
plt.figure(figsize =(16,9))
sns.heatmap(df.isnull())


# In[14]:


#Dropping the column 'society' because having high missing value percentage
df2  = df.drop('society',axis = 'columns')
df2.shape


# In[15]:


#filling the mean value in -->balcony feature
df2['balcony'] = df['balcony'].fillna(df2['balcony'].mean())
df2.isnull().sum()


# In[16]:


df3 = df2.dropna()
df3.isnull().sum().sum()


# In[17]:


df3.head()


# #  2. Feature Engineering

# In[18]:


#to show all rows and columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# Converting 'total_sqft' categorial feature into numeric

# In[19]:


df3['total_sqft'].value_counts()


# Here we observe that 'total_srft' contain string value in different format 
# 
# 
# -float,int like 2161.03,2663
# 
# -range value: 1070 - 1315
# 
# -number and string: 1100Sq. Meter, 132Sq. Yards
# 
# 
# Let's convert it into int format
# 

# In[20]:


total_sqft_int = []
for str_val in df3['total_sqft']:
    try:
        total_sqft_int.append(float(str_val))
        #if '123.4' like this convert this into float
    except:
        try:
            feet =[]
            feet = str_val.split('-')
            total_sqft_int.append((float(feet[0])+float(feet[-1]))/2)
            #Splitting the range value in two value and taking avg of it
        except:
            total_sqft_int.append(np.nan)
            #if value is not comtain in above then consider it as nan


# In[21]:


df4 = df3.reset_index(drop=True)
df4.head()


# In[22]:


df5 = df4.join(pd.DataFrame({'total_sqft_int':total_sqft_int}))
df5.head()


# In[23]:


df5.isnull().sum()


# In[24]:


df6 = df5.dropna()
df6.shape


# In[25]:


df6.info()


# #  Working On 'size' Feature

# In[26]:


df6['size'].value_counts()


# Let's assume that 2 BHK have 2 Bedroom and 2 RK which is
# 
# 
# 2BHK = 2Bedroom == 2RK
# 
# 
# so we takes only number and remove suffix txt

# In[27]:


size_int =[]
for str_val in df6['size']:
    room = []
    room = str_val.split(" ")
    try:
        size_int.append(int(room[0]))
    except:
        size_int.append(np.nan)
        print("Noice = ",str_val)
df6  = df6.reset_index(drop=True)


# In[28]:


df7 = df6.join(pd.DataFrame({'bhk':size_int}))
df7.shape


# In[29]:


def diag_plots(df,variable):
    plt.figure(figsize =(16,4))
    #Histogram
    plt.subplot(1,3,1)
    sns.distplot(df[variable],bins =30)
    plt.title("Histogram")
    #Q-Q plot
    plt.subplot(1,3,2)
    stats.probplot(df[variable],dist = "norm",plot =plt)
    plt.ylabel('Variable quantiles')
    #Boxplot
    plt.subplot(1,3,3)
    sns.boxplot(y = df[variable])
    plt.title("Boxplot")
    
    plt.show()
    


# In[30]:


num_var  = ["bath","balcony","total_sqft_int","bhk","price"]
for var in num_var:
    print("*****{}*****".format(var))
    diag_plots(df7,var)
    
    


# In[31]:


#Here we consider 1BHK requires 350 sqft
df7[df7['total_sqft_int']/df7['bhk']<350].head()


# In[32]:


df8 = df7[~(df7['total_sqft_int']/df7['bhk']<350)]
df8.shape


# In[33]:


df8['price_per_sqft']=df8['price']*100000/df8['total_sqft_int']
df8.head()


# In[34]:


#Removing Outliers
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean =np.mean(subdf.price_per_sqft)
        std = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(mean - std))&(subdf.price_per_sqft<=(mean+std))]
        df_out = pd.concat([df_out, reduced_df],ignore_index = True)
    return df_out

df9 = remove_pps_outliers(df8)
df9.shape


# In[35]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location)&(df.bhk==2)]
    bhk3 = df[(df.location==location)&(df.bhk==3)]
    plt.figure(figsize= (16,9))
    plt.scatter(bhk2.total_sqft_int, bhk2.price,color ='Blue',label = '2BHK', s=50)
    plt.scatter(bhk3.total_sqft_int, bhk3.price,color ='Red',label = '3BHK', s=50, marker = '+')
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.legend()
    
plot_scatter_chart(df9,"Rajaji Nagar")


# In the above scatter plot we observe that at same location price of 2bhk house is greater than 3bhk so it is consider as outlier

# In[36]:


# Removing BHK outliers
def remove_bhk_outliers(df):
  exclude_indices = np.array([])
  for location, location_df in df.groupby('location'):
    bhk_stats = {}
    for bhk, bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean':np.mean(bhk_df.price_per_sqft),
          'std':np.std(bhk_df.price_per_sqft),
          'count':bhk_df.shape[0]}
    for bhk, bhk_df in location_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices, axis='index')
 
df10 = remove_bhk_outliers(df9)
df10.shape


# In[37]:


plot_scatter_chart(df10,"Hebbal")


# In[38]:


df10.bath.unique()


# In[39]:


df10[df10.bath>df10.bhk+2]


# In[40]:


df11 = df10[df10.bath<df10.bhk+2]
df11.shape


# In[41]:


plt.figure(figsize=(16,9))
for i,var in enumerate(num_var):
    plt.subplot(3,2,i+1)
    sns.boxplot(df11[var])


# # Categorial Variable Encoding

# In[42]:


df11.head()


# In[43]:


df12 = df11.drop(['size','total_sqft'],axis =1)
df12.head()


# In[44]:


df13 = pd.get_dummies(df12,drop_first =True, columns = ['area_type','availability','location'])
df13.shape


# In[45]:


df13.head()


# In[46]:


df13.to_csv("oh_encoded_data.csv",index=False)


# # Working on 'area_type' Feature

# In[47]:


df12['area_type'].value_counts()


# In[48]:


df14 = df12.copy()
#Apllying One Hot Encoding on area_type
for cat_var in ['Super built-up','Built-up  Area','Plot  Area ']:
    df14["area_type"+cat_var]=np.where(df14['area_type']==cat_var,1,0)

df14.shape


# In[49]:


df14.head(2)


# # Working on 'availability' Feature 

# In[50]:


df14['availability'].value_counts()


# In[51]:


df14["availability_ready_to_move"] = np.where(df14["availability"]=="Ready To Move",1,0)
df14.shape


# # Working on 'location' Feature

# In[52]:


location_value_count = df14['location'].value_counts()
location_value_count


# In[53]:


location_gret_20 = location_value_count[location_value_count>=20].index

location_gret_20


# In[54]:


location_gret_20_ =['Whitefield', 'Sarjapur  Road', 'Electronic City', 'Marathahalli',
       'Raja Rajeshwari Nagar', 'Haralur Road', 'Hennur Road',
       'Bannerghatta Road', 'Uttarahalli', 'Thanisandra',
       'Electronic City Phase II', 'Hebbal', '7th Phase JP Nagar', 'Yelahanka',
       'Kanakpura Road', 'KR Puram', 'Sarjapur', 'Rajaji Nagar',
       'Kasavanhalli', 'Bellandur', 'Begur Road', 'Kothanur', 'Banashankari',
       'Hormavu', 'Harlur', 'Akshaya Nagar', 'Jakkur',
       'Electronics City Phase 1', 'Varthur', 'Chandapura', 'Ramamurthy Nagar',
       'Hennur', 'HSR Layout', 'Kundalahalli', 'Ramagondanahalli',
       'Koramangala', 'Kaggadasapura', 'Budigere', 'Hoodi', 'Hulimavu',
       'Malleshwaram', '8th Phase JP Nagar', 'Gottigere', 'JP Nagar',
       'Yeshwanthpur', 'Hegde Nagar', 'Channasandra', 'Bisuvanahalli',
       'Vittasandra', 'Indira Nagar', 'Kengeri', 'Old Airport Road',
       'Vijayanagar', 'Hosa Road', 'Brookefield', 'Sahakara Nagar',
       'Bommasandra', 'Green Glen Layout', 'Balagere', 'Old Madras Road',
       'Kudlu Gate', 'Rachenahalli', 'Panathur', 'Talaghattapura',
       'Thigalarapalya', 'Jigani', 'Kadugodi', 'Ambedkar Nagar', 'Mysore Road',
       'Yelahanka New Town', 'Attibele', 'Frazer Town', 'Devanahalli',
       'Dodda Nekkundi', 'Kanakapura', '5th Phase JP Nagar', 'TC Palaya',
       'Ananth Nagar', 'Lakshminarayana Pura', 'Nagarbhavi', 'Anekal',
       'Jalahalli', 'CV Raman Nagar', 'Kudlu', 'Kengeri Satellite Town',
       'Subramanyapura', 'Bhoganhalli', 'Kalena Agrahara', 'Horamavu Agara',
       'Doddathoguru', 'Hebbal Kempapura', 'Hosur Road', 'BTM 2nd Stage',
       'Vidyaranyapura', 'Domlur', 'Tumkur Road', 'Horamavu Banaswadi',
       'Mahadevpura']


# In[55]:


df15 = df14.copy()
for cat_var in location_gret_20_:
    df15['location'+cat_var] = np.where(df15['location']==cat_var,1,0)

df15.shape


# In[56]:


df15.head()


# In[57]:


df16 = df15.drop(['availability','location','area_type'],axis =1)
df16.shape


# # Split dataset in train and test

# In[58]:


X = df16.drop("price",axis=1)
Y = df16["price"]
print("Shape of X", X.shape)
print("Shape of Y",Y.shape)


# In[59]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state = 51)
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)
print("Shape of Y_train",Y_train.shape)
print("Shape of Y_test",Y_test.shape)


# # Machine Learning Model Training

# # >>Linear Regression<<

# In[60]:


lr = LinearRegression()
lr_lasso = Lasso()


# In[61]:


def rmse(Y_test,Y_pred):
    return np.sqrt(mean_squared_error(Y_test,Y_pred))


# In[62]:


#Linear Regression
lr.fit(X_train,Y_train)
lr_score = lr.score(X_test,Y_test)
lr_rmse = rmse(Y_test,lr.predict(X_test))
lr_score,lr_rmse


# In[63]:


#Lasso
lr_lasso.fit(X_train,Y_train)
lr_lasso_score = lr_lasso.score(X_test,Y_test)
lr_lasso_rmse = rmse(Y_test,lr_lasso.predict(X_test))
lr_lasso_score,lr_lasso_rmse


# # >>Support Vector Machine<<

# In[64]:


svr = SVR()
svr.fit(X_train,Y_train)
svr_score =svr.score(X_test,Y_test)
svr_rmse = rmse(Y_test,svr.predict(X_test))
svr_score,svr_rmse


# # >>Random Forest Regressor<<

# In[65]:


rfr = RandomForestRegressor()
rfr.fit(X_train,Y_train)
rfr_score= rfr.score(X_test,Y_test)
rfr_rmse = rmse(Y_test,rfr.predict(X_test))
rfr_score,rfr_rmse


# # >>XGBoost<<

# In[66]:


get_ipython().system('pip3 install xgboost')


# In[67]:


xgbr = XGBRegressor(verbosity = 0)
xgbr


# In[68]:


xgbr.fit(X_train,Y_train)
xgbr_score = xgbr.score(X_test,Y_test)
xgbr_rmse =rmse(Y_test, xgbr.predict(X_test))
xgbr_score, xgbr_rmse


# In[69]:


print(pd.DataFrame([{'Model':'Linear Regression','Score':lr_score,'RMSE':lr_rmse},
                   {'Model':'Lasso','Score':lr_lasso_score,'RMSE':lr_lasso_rmse},
                   {'Model':'Support Vector Machine','Score':svr_score,'RMSE':svr_rmse},
                   {'Model':'Random Forest Regressor','Score':rfr_score,'RMSE':rfr_rmse},
                   {'Model':'XGBoost','Score':xgbr_score,'RMSE':xgbr_rmse}],
                  columns = ['Model','Score','RMSE']))


# In[70]:


#Cross Validation for random forest regressor
cvs = cross_val_score(rfr,X_train,Y_train,cv =10)
cvs,cvs.mean()


# In[71]:


#Cross Validation for xgboost
cvs = cross_val_score(xgbr,X_train,Y_train,cv =10)
cvs,cvs.mean()


# In[72]:


np.sqrt(mean_squared_error(Y_test,rfr.predict(X_test)))


# In[73]:


np.sqrt(mean_squared_error(Y_test,xgbr.predict(X_test)))


# As you can see XGboost is giving better score and rmse than Random Forest Regressor
