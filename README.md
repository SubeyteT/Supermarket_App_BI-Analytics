# Supermarket_Chain_BI-Analytics

#### Aim of the project:

### Dataset:
#### Variables:





Head:

![image](https://user-images.githubusercontent.com/83431435/135341512-57a88b99-8c1e-4174-8ba1-4b737cf9497c.png)

### Data Preprocessing:

Missing Values:
![image](https://user-images.githubusercontent.com/83431435/135341570-50db421d-d872-41f7-adb8-61a7646f0461.png)

NA: Only Date_Order has NA's. Since we cannot fill date with this data we choose deleting NA rows.

Quantiles:

![image](https://user-images.githubusercontent.com/83431435/135341608-c260838e-544b-4c33-abfe-409517fa4029.png)

Number of categorical, numerical, categoric but cardinals, numerical but categorical columns:

Descriptive statistics for numeric columns:

![image](https://user-images.githubusercontent.com/83431435/135341759-d1e3024f-e62c-41d0-99f6-2c484115185b.png)

Distribution of classes of categorical columns (First two of them):
![image](https://user-images.githubusercontent.com/83431435/135341794-91b0ed38-cd60-4f27-a1e9-72cb8f6ac1ec.png)

Although some of the numerical columns seem to have outliers, large numbers of shopping is common in retail industry. When we look for the amounts bought per a transaction, what we see is the amounts are not outliers.![image](https://user-images.githubusercontent.com/83431435/135341839-b59ad362-85ca-4033-bf1f-651fd5894fb1.png)


### New Variables:

Recency: Number of days between customer's first and last purchase

T: Customer first alive – age from first date of order

Frequency: Customer's total number of transactions

Monetary: Money interaction expected from the customer

NEW_Asleep_Days: How many days the customer has not made a purchase.

NEW_Order_Interval: Customer's average shopping day range

NEW_day_of_month: What day of the month is the day of the transaction?

NEW_day_of_week: What day of the week is the day of the transaction?

NEW_is_wknd: Is the day of the transaction on the weekend?

NEW_is_month_start: Is the day of the transaction the beginning of the month?

NEW_is_month_end: Is the day of the transaction the end of the month?

NEW_Sale_Rate: Total discount rate per transaction

NEW_is_cancel: Has the order been completely cancelled?

NEW_T_Cat: Customer age categories (Oldest, Old, Med, New)

NEW_Pay_Per_Unit: Amount paid per unit on each transaction

Target: Target variable. Has the person continued his shopping habit?


### Missing Values:
It has been observed that there is NA in some values due to division by 0 while creating a variable. Since these values are not structural, they are filled with 0. 
Order range has NAs: It is due to customers who do not shop for the 2nd time.
![image](https://user-images.githubusercontent.com/83431435/135341459-9a302d0a-4f77-4d00-bd1a-031332a6ccad.png)

