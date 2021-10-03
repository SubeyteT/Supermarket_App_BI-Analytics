# Supermarket_Chain_BI-Analytics
 
## Business Problem:

An online supermarket chain application wants to know if first time shopping customers are going to shop for the second time. 

The dataset includes 6 months' transaction data. It is not singularized for each customer, which means there might be multiple transaction information of each.

### Variables:
*Customer_ID:* ID number of a customer
*Order_No:* Order number
*Date_Order:* Date of the order
*Order_DeliveryDate:* Delivery date of the order
*Membership_Date:* Membership date of the customer (we dropped, don't need it)
*UN_SER_Group:* EU Socio Economic Ratio Ranges of the Neighborhood of the Address to which the Order is Delivered
*Income_Level:* Monthly Household Income Level Ranges of the Neighborhood where the Order is Delivered
*Type_Delivered_Product:* Number of Different Items Delivered
*Amount_ofDelivered_Product:* Total Items Delivered
*NetPrice_Delivered:* Amount of Customer Expenditure for Delivered Products (shipping fee is not included in this amount)
*Type_Cancelled_Prod:* Different Product Quantity Ordered But Not Received
*Amount_Cancelled_Prod:* Total Items Ordered But Not Received
*NetPrice_Cancelled:* Total Amount of Items Ordered But Not Received (shipping is not included in this amount)
*Total_Sale:* Total Discount Provided in the Order (It is the total of the discounts distributed throughout the cart)
*Total_Shipping_Fee:* Order Shipping Fee
*Sect_1..Sect_10:* Indicates whether the customer has placed an order from the relevant section.

### Exploratory Data Analysis
Head:
![image](https://user-images.githubusercontent.com/83431435/135758680-5e6edad4-6ce0-43a9-bff7-ff4503559791.png)

NA: Only Date_Order has NA's. Since we cannot fill date with this data we choose deleting NA rows.
![image](https://user-images.githubusercontent.com/83431435/135758693-956b6614-57df-4ce7-8970-70af895e5a1e.png)

Number of categorical, numerical, categoric but cardinals, numerical but categorical columns:
![image](https://user-images.githubusercontent.com/83431435/135758713-fe71be3b-8453-4286-934e-16938b2164ef.png)

Descriptive statistics for numeric columns:
![image](https://user-images.githubusercontent.com/83431435/135758732-5f02cc3a-0afd-4065-87dc-47672c5dec25.png)

Distribution of classes of categorical columns (First two of them):
![image](https://user-images.githubusercontent.com/83431435/135758747-3a26009a-78ea-4d2f-9bb4-f0329ca81f10.png)

Although some of the numerical columns seem to have outliers, large numbers of shopping is common in retail industry. When we look for the amounts bought per a transaction, what we see is the amounts are not outliers. For example, while 2493 is the hisghest record as an outlier in the variable of NetPrice_Delivered, since products at this price were also found in practice, no action was taken.
![image](https://user-images.githubusercontent.com/83431435/135758764-90bdbdff-9216-46be-8cad-0751a936640a.png)

### New Variables:

*NEW_Asleep_Days:* Number of days between customer's first and second purchase. It is going to be used only for train-test dfs separation.
*NEW_Days_Interval:* How many days until the second purchase? 
-----Some customers has transactions more than 2 times. According to our aim, we are predicting only if they will shop for the second time or no, so we only need first two transctions of each customer. From this point we don't need the rest observations so, dropping is done.
*NEW_day_of_week:* What day of the week is the day of the transaction?
*NEW_is_wknd:* Is the day of the transaction on the weekend?
*NEW_Sale_Rate:* Total discount rate per transaction
*NEW_is_cancel:* Has the order been completely cancelled?
*NEW_Pay_Per_Unit:* Amount paid per unit on each transaction
*Target:* Target variable. Has the person continued his shopping habit?

It has been observed that there is NA in some values due to division by 0 while creating a variable. Since these values are not structural, they are filled with 0.
![image](https://user-images.githubusercontent.com/83431435/135758800-9898a06d-fa56-434e-9d37-d2143a59b9a2.png)

## Target Analysis

The dataset doesn't involve target variable, so I had to set it up. Since our aim is to find if a customer is going to purchase for the second time or not,  the target variable has to give a hint from the older data. I assigned "1" for the customers who has more than 1 transactions and "0" for the first timers. 

Now that there are features about the first and second transactions' relationship, we only need the first transction of each customer. So, singularization of Customer_ID feature is done, keeping the first transaction.

## Encoding

#### Rare Encoding:
First I have started with analysing the rare classes of categorical variables. I have used my rare_analyser function for this purpose. There is no need for rare encoding since there aren't any rare classes. 

#### Label & One-Hot Encoding:
Among the categorical variables only 3 variables ['UN_SER_Group', 'Income_Level', 'NEW_day_of_week'] need to be encoded. They all have more than 2 classes, so one-hot encoding is enough. After this operation, there are 47 variables. 

Now there might be some useless variables. Classes having less than %1 frequency may lead us to consider them as useless. Also, there is no need for variables like Customer ID, Order date beacuse, we earned information using them, machine learning algorithms don't learn anything from them. So above mentioned columns are dropped.

## LGBM

For allocating test and train dfs, I have found average days until second transaction, which is 22 days. This implies that if a customer doesn't have any transaction in last 22 days, she can be considered as churn and has a target value of "0".  Train_df includes observations until the last 22 days, without the "NEW_Asleep_Days" column, we don't need it anymore. Test_df includes observations in the last 22 days ans targets = "0". Because we want to know if they will shop again or not, we already know the rest. 

Also %20 of the train set is splitted for validation dataset. 

LGBM is preffered for ML algorithm because of its fast and successful performance. 

The results are: np.sqrt of  y_pred(X_train) = 0.0927
                         np.sqrt of  y_pred(X_val) = 0.0969

####  Feature Importance and Selection

I have used the plot_importance func. for this operation. Since there aren't any non effective features for the model, selected columns include all the variables and we keep going with all of them.
![image](https://user-images.githubusercontent.com/83431435/135758865-7e90e0cd-d3db-426a-8063-8ae8a154cbba.png)

#### Hyperparameter Optimization with Selected Features

rmse = 0.1045

## Logistic Regression

Also, I have used the logistic regression model for prediction. The results are given with the plots below.
![image](https://user-images.githubusercontent.com/83431435/135758881-1f0be2ee-7df0-4d05-8ce9-60bc7b85e0a2.png)
![image](https://user-images.githubusercontent.com/83431435/135758888-8b5b6e93-eddf-4097-a0f0-56975518a0ef.png)
![image](https://user-images.githubusercontent.com/83431435/135758896-bf1b1ff5-9829-408c-b86c-d34622039ba9.png)
![image](https://user-images.githubusercontent.com/83431435/135758900-31c70c13-178f-4aad-aad5-73caf63c038f.png)







