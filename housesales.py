#!/usr/bin/env python
# coding: utf-8

# ##  1.Data Loading and Exploration

# ### Necessary libraries are imported, and the dataset (kc_house_data.csv) is loaded using Pandas.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r"C:\Users\Desktop\amazon\house-price-prediction-master\house-price-prediction-master\kc_house_data.csv")


# ###  The head() function shows the first few rows of the dataset, and describe() provides summary statistics.

# In[3]:


data.head()


# In[4]:


data.describe()


# ## 2. Data Visualization 

# ### This code generates a bar plot showing the distribution of the number of bedrooms in the dataset.

# In[5]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# ### This part creates a joint plot to visualize the geographical distribution of houses. Latitude and Longitude are used for the x and y axes, respectively.

# In[6]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine


# ### These scatter plots explore the relationship between the house price and various features like square footage, location, bedrooms.

# In[7]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[8]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[9]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[10]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[11]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[12]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# # 3. Feature Engineering and Data Preparation

# ###  Features are selected for training the model. The bar plot and scatter plots explore the relationship between different features and house prices.

# In[13]:


train1 = data.drop(['id', 'price'],axis=1)


# In[14]:


train1.head()


# In[15]:


data.floors.value_counts().plot(kind='bar')


# In[16]:


plt.scatter(data.floors,data.price)


# In[17]:


plt.scatter(data.condition,data.price)


# In[18]:


plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


# #  Linear Regression Model

# ### Linear Regression model is trained using the features and labels. The model's performance is evaluated using the R-squared score.

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


reg = LinearRegression()


# In[21]:


labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[24]:


reg.fit(x_train,y_train)


# In[25]:


reg.score(x_test,y_test)


# In[51]:


y_pred_linear = reg.predict(x_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression - Root Mean Squared Error: {rmse_linear}")


# # Gradient Boosting Regressor Model

# ### Gradient Boosting Regressor model is trained and evaluated on the same dataset.calculates the test error at each boosting stage for the Gradient Boosting Regressor model.

# In[58]:


from sklearn.ensemble import GradientBoostingRegressor


# In[26]:


clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                learning_rate=0.1, loss='squared_error')
clf.fit(x_train, y_train)


# In[27]:


clf.score(x_test,y_test)


# In[52]:


y_pred_gb = clf.predict(x_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print(f"Gradient Boosting Regressor - Root Mean Squared Error: {rmse_gb}")


# In[29]:


params = {
    'n_estimators': 400,
    'max_depth': 5,
    'min_samples_split': 2,
    'learning_rate': 0.1,
    'loss': 'squared_error'
}

t_sc = np.zeros((params['n_estimators']), dtype=np.float64)


# In[30]:


y_pred = reg.predict(x_test)


# In[33]:


from sklearn.metrics import mean_squared_error
t_sc = np.zeros((clf.n_estimators,), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i] = mean_squared_error(y_test, y_pred)

testsc = np.arange(clf.n_estimators) + 1


# In[34]:


testsc = np.arange((params['n_estimators']))+1


# In[35]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[50]:


t_sc = np.zeros((clf.n_estimators,), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i] = mean_squared_error(y_test, y_pred)

best_stage = np.argmin(t_sc) + 1  # Adding 1 to convert from zero-based index to stage number
best_mse = t_sc[best_stage - 1]  # Subtracting 1 to get the correct index

print(f"Best Boosting Stage: {best_stage}")
print(f"Mean Squared Error at Best Stage: {best_mse}")


# In[36]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[37]:


pca = PCA()


# In[38]:


pca.fit_transform(scale(train1))


# In[39]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


# #  XGBoost model 

# ### XGBoost model is trained using the XGBRegressor. Predictions are made on the test set.

# In[43]:


import xgboost as xgb


# In[44]:


# XGBoost Regressor
params = {'objective': 'reg:squarederror', 'colsample_bytree': 0.3, 'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10, 'n_estimators': 400}


# In[45]:


xg_reg = xgb.XGBRegressor(**params)
xg_reg.fit(x_train, y_train)


# In[46]:


# Predictions
y_pred = xg_reg.predict(x_test)


# In[57]:


from sklearn.metrics import r2_score


# In[56]:


r2_xgb = r2_score(y_test, y_pred)
print(f"XGBoost - R-squared score: {r2_xgb}")


# In[53]:


# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost - Root Mean Squared Error: {rmse}")


# In[48]:


# Feature Importance
xgb.plot_importance(xg_reg)
plt.show()


# In[ ]:




