# Supermarket_Chain_BI-Analytics

Head:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3f42d1f9-c974-432b-9fcb-a4bf7d64697d/Untitled.png)

NA: Only Date_Order has NA's. Since we cannot fill date with this data we choose deleting NA rows.

Quantiles:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4e8d45e1-aeeb-43d1-81bb-c9cf25964939/Untitled.png)

Number of categorical, numerical, categoric but cardinals, numerical but categorical columns:

Descriptive statistics for numeric columns:

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2fae5eae-cfc5-4f14-a984-fd2c545d09eb/Untitled.png)

Distribution of classes of categorical columns (First two of them):

Although some of the numerical columns seem to have outliers, large numbers of shopping is common in retail industry. When we look for the amounts bought per a transaction, what we see is the amounts are not outliers.

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

It has been observed that there is NA in some values due to division by 0 while creating a variable. Since these values are not structural, they are filled with 0. 
Order range has NAs: It is due to customers who do not shop for the 2nd time.
