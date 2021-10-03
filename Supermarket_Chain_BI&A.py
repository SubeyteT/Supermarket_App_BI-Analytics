########################
# Supermarket Chain BI & Analytics Case
########################

import pandas as pd
import numpy as np
import seaborn as sns
from helpers.data_prep import *
from helpers.eda import *
import datetime as dt
import missingno as msno
from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score, \
    train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, \
    mean_squared_error, r2_score, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 170)

########################
# EDA
########################

df_ = pd.read_csv("datasets/Supermarket_Chain.txt", sep=",")
df = df_.copy()
df.head()

check_df(df)

# checking missing values:
missing_values_table(df)
# only date has NAs
df.dropna(inplace=True)

# turning dtype of dates into datetime:
df["Date_Order"] = (df["Date_Order"]).astype(int)
df['Date_Order'] = pd.to_datetime(df['Date_Order'], format = "%Y%m%d")
df['Order_DeliveryDate'] = pd.to_datetime(df['Order_DeliveryDate'], format = "%Y%m%d")
df['Membership_Date'] = pd.to_datetime(df['Membership_Date'], format = "%Y%m%d")
df.drop("Membership_Date", axis=1, inplace=True)

df =df.sort_values(by='Date_Order', ascending=False)

########################
# EDA
########################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerical variable analysis:
df[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]).T

for col in df[num_cols]:
    print(col, num_summary(df, col))

# Categorical variable analysis:
for col in df[cat_cols]:
    print(col, cat_summary(df, col))

#######################
# Outlier Analysis
#######################

for col in df[num_cols]:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

for col in df[num_cols]:
    print(col, outlier_thresholds(df, col, q1=0.01, q3=0.99))

########################
# FEATURE EXTRACTION
########################

df['Date_Order'].max()
today_date = dt.datetime(2021, 2, 2) # 2 days after the last transaction in dataset for recency calculation
date_df = df.groupby('Customer_ID').agg({'Date_Order': lambda date: (today_date - date.max()).days})
date_df.columns = ['NEW_Asleep_Days']
df = pd.merge(df, date_df, on="Customer_ID")

# we will keep with first two orders of customers
df = df.groupby('Customer_ID').head(2)

# How many days until second transaction?
Days_second = df.groupby('Customer_ID').agg({'Date_Order': lambda date: (date.max() - date.min()).days})
Days_second.columns = ['NEW_Days_Interval']
df = pd.merge(df, Days_second, on="Customer_ID")

Days_second.head()

df['NEW_Days_Interval'].value_counts()

# time intervals:
def create_date_features(df):
    df['NEW_day_of_week'] = df.Date_Order.dt.dayofweek + 1
    df['NEW_is_wknd'] = df.Date_Order.dt.weekday // 4
    return df

create_date_features(df)

# indirim oranı
df['NEW_Sale_Rate'] = df['Total_Sale'] / (df['Total_Sale'] + df['NetPrice_Delivered'] + df['NetPrice_Cancelled'])

# If all of the order is cancelled:
df.loc[(df['Amount_ofDelivered_Product'] == 0), "NEW_is_cancel"] =  int(1)
df.loc[(df['Amount_ofDelivered_Product'] != 0), "NEW_is_cancel"] =  int(0)

# shipping fee categorical
df['Total_Shipping_Fee'] = [1 if x > 0 else 0 for x in df['Total_Shipping_Fee']]

## Average pay per unit:
df['NEW_Pay_Per_Unit'] = df['NetPrice_Delivered'] / df['Amount_ofDelivered_Product']

missing_values_table(df)
df.fillna(0, inplace=True)

########################
# TARGET ANALYSIS
########################

# Number of unique customers
df['Customer_ID'].nunique() # 45838

######## Target Variable:
# customers with multiple purchases :1
# First time customers: 0

df['Counts'] = df.groupby(['Customer_ID'])['Order_No'].transform('count')
df['Counts'].head()

df['Target'] = [1 if  df['Counts'][i] > 1 else 0 for i in range(0,len(df['Customer_ID']))]
df['Target'].value_counts()

df.drop('Counts', axis=1, inplace=True)

# SINGULARIZATION OF Customer_ID:
df.drop_duplicates(subset='Customer_ID', keep='first', inplace=True)
df.shape

#######################################
# ENCODING
#######################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

### Rare Encoding:

rare_analyser(df, 'Target', cat_cols)

rare_encoder(df, 0.01)

### Label & One-Hot Encoding:

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape  ## 47 variables
# There might be trash variables

cat_cols, num_cols, cat_but_car = grab_col_names(df)

check_df(df)

rare_analyser(df, "Target", cat_cols)

# useless cols:
useless_cols_new = [col for col in cat_cols if (df[col].value_counts()
                                                 / len(df) <= 0.01).any(axis=None)]

# Drop other useless columns
drops = ['Date_Order', 'Order_DeliveryDate','Order_No', 'Customer_ID']
drop = drops + useless_cols_new
df = df.drop(drops, axis=1)
df.columns
df.shape #44

#######################################
# MODELING
#######################################

########### train set:
# average days until second transaction: 22 days
df['NEW_Days_Interval'].sum() / len(df[df['Target'] == 1])

# if a customer has shopped in last 22 days :Train dataset, else: Test dataset

train_df = df.loc[~(df['NEW_Asleep_Days'] < 22)].drop('NEW_Asleep_Days', axis=1)
train_df.shape # (34797, 44)
df.shape

test_df = df.loc[(df['NEW_Asleep_Days'] < 22) & (df['Target'] == 0)].drop(['Target', 'NEW_Asleep_Days'], axis=1)
test_df.shape # 3636 customers

X = train_df.drop(['Target'], axis=1)
y = train_df['Target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=22)

# LGBM
lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01,0.05,1],
               "n_estimators": [1000,1500,6000,8000,15000],
               "max_depth": [-1,2,5],
               "colsample_bytree": [0.50,1,2],
               "num_leaves": [5,7,15,20,30]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=True).fit(X_train, y_train)

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
#Best params: {'colsample_bytree': 1,'learning_rate': 0.01,'max_depth': -1,'n_estimators': 8000,'num_leaves': 7}
#Best params: {'colsample_bytree':0.5,'learning_rate': 0.01,'max_depth': 2,'n_estimators': 8000,'num_leaves': 5}

y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 0.08682191769540416 , 0.09272575390105069
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred)) # 0.09755088359259019 , 0.09695695518837591


#######################################
# Feature Importance
#######################################
def plot_importance(model, df, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': df.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=True)[0:num])
    plt.title('df')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_tuned, X_train)

df.columns

df.head()

#######################
# Feature Selection
#######################

feature_imp = pd.DataFrame({'Value': lgbm_tuned.feature_importances_, 'Feature': X.columns})
feature_imp[feature_imp["Value"] > 0].shape

feature_imp[feature_imp["Value"] == 0].shape

zero_imp_cols = feature_imp[feature_imp["Value"] == 0]["Feature"].values
selected_cols = [col for col in X.columns if col not in zero_imp_cols]
len(selected_cols)

# there aren't any noneffective features for the model.

#######################################
# Hyperparameter Optimization with Selected Features
#######################################

lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.5],
               "n_estimators": [8000, 7500],
               "colsample_bytree": [0.5, 0.7, 0.3]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
# rmse: 0.1045619743778253

#######################################
# Target Predicting for Test Dataset
#######################################

submission_df = pd.DataFrame()
y_pred_sub = final_model.predict(test_df)
y_pred_sub = np.expm1(y_pred_sub)
submission_df['Target'] = y_pred_sub
submission_df.head()

#######################################
# LOGISTIC REGRESSION
#######################################

###################
# SCALING
###################
df.head()

df_robust = df.loc[~(df['NEW_Asleep_Days'] < 22)]
cat_cols, num_cols, cat_but_car = grab_col_names(df_robust)

scaler = RobustScaler()
df_robust[num_cols] = scaler.fit_transform(df_robust[num_cols])
df_robust[num_cols].head()

check_df(df_robust)

############ Model:

X_l = df_robust.drop(['Target'], axis=1)
y_l = df_robust['Target']
X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(X_l, y_l, test_size=0.20, random_state=22)

log_model = LogisticRegression().fit(X_train_l, y_train_l)
log_model.intercept_
log_model.coef_

############ Prediction:
y_pred_l = log_model.predict(X_train_l)

# Train Accuracy
y_pred_l = log_model.predict(X_train_l)
accuracy_score(y_train_l, y_pred_l) # 0.9890074361461364

### Test
# AUC Score için y_prob
y_prob_l = log_model.predict_proba(X_val_l)[:, 1]

# Diğer metrikler için y_pred
y_pred_l = log_model.predict(X_val_l)

# ACCURACY
accuracy_score(y_val_l, y_pred_l)    # 0.9892241379310345

# PRECISION
precision_score(y_val_l, y_pred_l)   # 1

# RECALL
recall_score(y_val_l, y_pred_l)     # 0.967032967032967

# F1
f1_score(y_val_l, y_pred_l)    # 0.9832402234636871

# CONFUSION MATRIX
confusion_matrix(y_val_l,y_pred_l)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_val_l, y_pred_l)


# ROC CURVE
plot_roc_curve(log_model, X_val_l, y_val_l)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_val_l, y_prob_l)   # 0.9918404071915278

# Classification report
print(classification_report(y_val_l, y_pred_l))

