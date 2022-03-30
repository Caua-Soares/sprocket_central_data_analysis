#%%
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load in the provided data
transactions = pd.read_excel(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\KPMG_VI_New_raw_data_update_final (Modified).xlsx", sheet_name="Transactions")
new_cust = pd.read_excel(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\KPMG_VI_New_raw_data_update_final (Modified).xlsx", sheet_name="NewCustomerList")
cust_dem = pd.read_excel(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\KPMG_VI_New_raw_data_update_final (Modified).xlsx", sheet_name="CustomerDemographic")
cust_add = pd.read_excel(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\KPMG_VI_New_raw_data_update_final (Modified).xlsx", sheet_name="CustomerAddress")

# Create a copy of our original data so that we can always go back to original
trans_mod = transactions.copy()
newcust_mod = new_cust.copy()
custdem_mod = cust_dem.copy()
custadd_mod = cust_add.copy()

# We will load in some geographical data that will help us make sense of postcodes
aus_geo = pd.read_excel(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\australian_postcodes.xlsx")
ausgeo_mod = aus_geo.copy()

# We need to rename column 'Postcode' to 'postcode' in order to correctly join later on
ausgeo_mod = ausgeo_mod.rename(columns={'Postcode': 'postcode'})
# We only need 4 variables from this dataset: postcode, SA 4 Name, Long and Lat
ausgeo_mod = ausgeo_mod[['postcode', 'SA4 Name', 'Long', 'Lat', 'MMM 2019']]
# Since we only want this data to roughly understand where customers are from, we do not need extremely precise geolocations for these customers
# As long as we know the town and have at least one longitude and latitude coordinates, it should be enough for our purposes
# Therefore, we will drop duplicated postcodes. This will make it easier to use in our dataset later on
ausgeo_mod = ausgeo_mod.drop_duplicates(subset='postcode')


'''
This section focus on exploring Customer Demographics data
'''
# The first thing we want to do is to observe if there is any age pattern in our current customers
# In order to do that, we must create a new column that represents the age of each customer from DOB to today
custdem_mod['DOB'] = pd.to_datetime(custdem_mod['DOB']).apply(lambda x: x.date())
# Additionally, there is a mistake on index 33, shown to be born in 1843 meaning 177 years old.
custdem_mod.at[33, 'DOB'] = pd.to_datetime('1943-12-21')

# Agriculture is misspelled (Argiculture) in job_industry_category, so we need to replace with the correct spelling
custdem_mod['job_industry_category'] = custdem_mod['job_industry_category'].replace(to_replace='Argiculture', value='Agriculture')

# There are 87 customers missing DOB, since this represents only a 2.2% of our dataset, we will exclude it
# custdem_mod = custdem_mod.drop(custdem_mod[custdem_mod['DOB'].isna()].index)
# Drop column 'default' because it does not mean anything for us
custdem_mod = custdem_mod.drop(columns='default')
# Drop the empty columns in transaction
trans_mod = trans_mod.dropna(axis=1, how='all')

# This function calculates the age of each customer
def from_dob_to_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
# Now we can create the new column using the function that gives us the age
custdem_mod['Age'] = custdem_mod['DOB'].apply(lambda x: from_dob_to_age(x))

# Following line characterizes 'Age' in bins and round it
hist_data = custdem_mod['Age'][~pd.isna(custdem_mod['Age'])]
# counts, bins = np.histogram(custdem_mod['Age'], bins=6)
counts, bins = np.histogram(hist_data, bins=6)
bins = np.round(bins, 0)
custdem_mod['Bin_Age'] = pd.cut(custdem_mod['Age'], bins=bins)

# plt.hist(custdem_mod['Age'], bins=6, edgecolor='black')
# plt.title('Customer Distribution by Age')
# plt.xlabel("Age Group")
# plt.ylabel("Frequency")
# plt.show()

# Let's create a 'balance' column in custdem_mod
s = trans_mod[trans_mod['customer_id'].isin(custdem_mod['customer_id'])].groupby('customer_id')['list_price'].sum()
custdem_mod['balance'] = round(custdem_mod['customer_id'].map(s), 2)

# Let's create a 'number_of_purchases' column in custdem_mod
c = trans_mod[trans_mod['customer_id'].isin(custdem_mod['customer_id'])].groupby('customer_id')['transaction_id'].count()
custdem_mod['number_of_purchases'] = round(custdem_mod['customer_id'].map(c), 0)



# This function help us properly label our next graph
def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x*1e-6)

formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)

# # Plots a bar chart describing the total sales volume per age interval
# age_group_tot_balance = custdem_mod.groupby('Bin')['balance'].sum()
# age_group_tot_balance.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.title('Sales Volume by Age')
# plt.xlabel("Age Group")
# plt.ylabel("Total Sales Volume")
# plt.tight_layout()
# plt.show()

# # Plots a bar chart describing the average balance per age interval
# age_group_tot_balance = custdem_mod.groupby('Bin')['balance'].mean()
# age_group_tot_balance.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.title('Average Customer balance by Age')
# plt.xlabel("Age Group")
# plt.ylabel("Average Customer balance")
# plt.tight_layout()
# plt.show()

# # Plots a bar chart describing the average number_of_purchases per age interval
# age_group_tot_balance = custdem_mod.groupby('Bin')['number_of_purchases'].mean()
# age_group_tot_balance.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Average Customer balance by Age')
# plt.xlabel("Age Group")
# plt.ylabel("Average Customer balance")
# plt.tight_layout()
# plt.show()

'''
The following section explores gender distribution of customers in relation to balance and number_of_purchases
'''

# We need to address the inconsistencies in 'gender'
custdem_mod['gender'] = custdem_mod['gender'].replace(['F', 'Femal'], 'Female')
custdem_mod['gender'] = custdem_mod['gender'].replace(['M', 'U'], 'Male')

# # Pie chart demonstrating the customer distribution per gender
# labels = custdem_mod['gender'].unique()
# explode = [0.02, 0.02]
# colors = ['plum', 'royalblue']
#
# fig1, ax1 = plt.subplots()
# ax1.pie(custdem_mod.groupby('gender')['gender'].count(), explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=False, startangle=90, colors=colors)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# ax1.set_title('Customer Gender Distribution')
# plt.show()

# # Plots a bar chart describing the total sale volume per gender
# test = custdem_mod.groupby('gender')['balance'].sum()
# test.plot(kind='bar')
#
# # Plots a bar chart describing the average sale volume per gender
# test = custdem_mod.groupby('gender')['balance'].mean()
# test.plot(kind='bar')
#
# # Plots a bar chart describing the average number_of_purchases per gender
# test = custdem_mod.groupby('gender')['number_of_purchases'].sum()
# test.plot(kind='bar')
#
# # Plots a bar chart describing the total number_of_purchases per gender
# test = custdem_mod.groupby('gender')['number_of_purchases'].mean()
# test.plot(kind='bar')

'''
This section explores the purchasing history in relation to customer balance and number_of_purchases
'''
# Following line characterizes 'past_3_years_bike_related_purchases' in bins and round it
counts_3purch, bins_3purch = np.histogram(custdem_mod['past_3_years_bike_related_purchases'], bins=6)
bins = np.round(bins_3purch, 0)
custdem_mod.insert(5, 'Bin_Purch', pd.cut(custdem_mod['past_3_years_bike_related_purchases'], bins=bins_3purch))

# # Plots a bar chart describing the average balance per age interval
# past_puch_tot_balance = custdem_mod.groupby('Bin_Purch')['balance'].sum()
# past_puch_tot_balance.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.title('Sales Volume by Past 3 Years Purchases')
# plt.xlabel("Past 3 Years Purchases")
# plt.ylabel("Sales Volume")
# plt.tight_layout()
# plt.show()
#
#
#
# # Plots a bar chart describing the average balance per age interval
# past_puch_mean_balance = custdem_mod.groupby('Bin_Purch')['balance'].mean()
# past_puch_mean_balance.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.title('Average Customer balance by Past 3 Years Purchases')
# plt.xlabel("Past 3 Years Purchases")
# plt.ylabel("Average Customer balance")
# plt.tight_layout()
# plt.show()
#
# # Plots a bar chart describing the average number_of_purchases per age interval
# past_puch_tot_purch = custdem_mod.groupby('Bin_Purch')['number_of_purchases'].sum()
# past_puch_tot_purch.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Total number_of_purchases by Past 3 Years Purchases')
# plt.xlabel("Past 3 Years Purchases")
# plt.ylabel("Total number_of_purchases")
# plt.tight_layout()
# plt.show()
#
# # Plots a bar chart describing the average number_of_purchases per age interval
# past_puch_mean_purch = custdem_mod.groupby('Bin_Purch')['number_of_purchases'].mean()
# past_puch_mean_purch.plot(kind='bar', figsize=(10, 5))
# plt.xticks(rotation=30)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Average number_of_purchases by Past 3 Years Purchases')
# plt.xlabel("Past 3 Years Purchases")
# plt.ylabel("Average number_of_purchases")
# plt.tight_layout()
# plt.show()

'''
The following section explores wealth in relation to balance and number_of_purchases
'''
# # Verify data in wealth segment column
# custdem_mod['wealth_segment'].unique()
#
# # Pie chart demonstrating the customer distribution per gender
# labels = custdem_mod['wealth_segment'].unique()
# explode = [0.02, 0.02, 0.02]
# colors = ['tab:red', 'cornflowerblue', 'limegreen']
#
# fig2, ax2 = plt.subplots()
# ax2.pie(custdem_mod.groupby('wealth_segment')['wealth_segment'].count(), explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=False, startangle=90, colors=colors)
# ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# ax2.set_title('Customer Wealth Segment Distribution')
# plt.show()

# # Plots a bar chart describing the total sale volume per wealth segment
# test = custdem_mod.groupby('wealth_segment')['balance'].sum()
# test.plot(kind='bar')
# plt.xticks(rotation=0)
# plt.title('Total Sale Volume by Wealth Segment')
# plt.xlabel("Wealth Segment")
# plt.ylabel("Total Sale Volume")
# plt.tight_layout()
# plt.show()
#
# # Plots a bar chart describing the average sale volume per wealth segment
# test = custdem_mod.groupby('wealth_segment')['balance'].mean()
# test.plot(kind='bar')
# plt.xticks(rotation=0)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Average Sale Volume by Wealth Segment')
# plt.xlabel("Wealth Segment")
# plt.ylabel("Average Sale Volume")
# plt.tight_layout()
# plt.show()
#
# # Plots a bar chart describing the average number_of_purchases per wealth segment
# test = custdem_mod.groupby('wealth_segment')['number_of_purchases'].sum()
# test.plot(kind='bar')
# plt.xticks(rotation=0)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Total number_of_purchases by Wealth Segment')
# plt.xlabel("Wealth Segment")
# plt.ylabel("Total number_of_purchases")
# plt.tight_layout()
# plt.show()
#
# # Plots a bar chart describing the total number_of_purchases per wealth segment
# test = custdem_mod.groupby('wealth_segment')['number_of_purchases'].mean()
# test.plot(kind='bar')
# plt.xticks(rotation=0)
# plt.ticklabel_format(axis='y', style='plain')
# plt.title('Average number_of_purchases by Wealth Segment')
# plt.xlabel("Wealth Segment")
# plt.ylabel("Average number_of_purchases")
# plt.tight_layout()
# plt.show()

"""
Now lets look at Customer Address data
"""
# The first thing I see is that State has discrepancies in the data (NSW vs New South Wales)
# Since state data for new customers is abbreviated, let's transform this data so they are in the same format
custadd_mod['state'] = custadd_mod['state'].replace(['New South Wales', 'Victoria'], ['NSW', 'VIC'])

# Let's add the balance and number_of_purchases column into custadd_mod to further analyze
custadd_mod = custadd_mod.join(custdem_mod.set_index('customer_id')['balance'], how='inner', on='customer_id')
custadd_mod = custadd_mod.join(custdem_mod.set_index('customer_id')['number_of_purchases'], how='inner', on='customer_id')

# # Check which customer ids are in transaction that are not in custdem_mod
# res = trans_mod[~trans_mod['customer_id'].isin(custdem_mod['customer_id'])]

# We want to add the town, the Longitude and Latitude to Custadd_mod
# We have to make sure that we are matching postcodes
custadd_mod = custadd_mod.merge(ausgeo_mod, how='inner', on='postcode')

# # Visualize MMM 2019 variable
# test = custadd_mod.groupby('MMM 2019')['balance'].sum()
# test.plot(kind='bar')
# plt.show()


"""
Let's look at the transaction data
"""
# the only modification we need to do is to transform the column 'product_first_sold_date' to datetime
# trans_mod['product_first_sold_date'] = pd.to_datetime(trans_mod['product_first_sold_date'])

trans_mod['profit_margin'] = trans_mod['list_price'] - trans_mod['standard_cost']

"""
This section I'm saving the modified dataframes into a excel spreadsheet so we can visualize using Power BI
"""
# Create a Pandas Excel Writer using XlsxWriter as the engine
writer = pd.ExcelWriter('Bike Data Modified.xlsx', engine='xlsxwriter')

# write each DataFrame to a specific sheet
trans_mod.to_excel(writer, sheet_name='Transactions')
custdem_mod.to_excel(writer, sheet_name='CustomerDemographic')
custadd_mod.to_excel(writer, sheet_name='CustomerAddress')
newcust_mod.to_excel(writer, sheet_name='NewCustomerList')

# Close the pandas excel writer and output the excel file
writer.save()

"""
In this section, we want to modify the variables so that we can use them to develop a model
"""
# First, let's bring all the desired variables into one dataframe: customer_id, gender, Bin_Purch, Bin_Age, job_industry, wealth_segment, owns_car, tenure, MMM 2019, State, property_valuation, balance and number_of_purchases
data = custdem_mod[['customer_id', 'gender', 'past_3_years_bike_related_purchases', 'Age', 'job_industry_category', 'wealth_segment', 'owns_car', 'tenure', 'balance', 'number_of_purchases']]
data = data.merge(custadd_mod[['customer_id', 'state', 'property_valuation', 'MMM 2019']], how='inner', on='customer_id' )

data = data.rename(columns={'past_3_years_bike_related_purchases': 'bike_related_purchases', 'job_industry_category': 'job_industry', 'MMM 2019': 'remoteness_level'})

# Now we must transform the following categorical variables into binary so that we can use them in our model: gender, job_industry_category, wealth_segment, owns_car, state
dummies = pd.get_dummies(data[['gender', 'job_industry', 'wealth_segment', 'owns_car', 'state']], drop_first=True)

# Concatenate dummies to data dataframe
data_final = pd.concat((data, dummies), axis=1)

# We must drop the original variables
data_final = data_final.drop(['gender', 'job_industry', 'wealth_segment', 'owns_car', 'state'], axis=1)

# Additionally, we must drop all nan
data_final = data_final.dropna(how='any')

# We want to create a classification variable that classifies whether the customer is a high valued customer or nor
# We will use the the mean of balance as the threshold

dict_hv = []
for i in data_final.index:
    if data_final['balance'][i] >= data_final['balance'].quantile(.75):
        dict_hv.append(1)
    else:
        dict_hv.append(0)

data_final['high_value_cust'] = dict_hv

# data_final.to_csv(r"C:\Users\Soares\Documents\Important\Practice\KPMG Virtual Internship\Data_Final.csv", index=False)
#
# # Create a Pandas Excel Writer using XlsxWriter as the engine
# writer = pd.ExcelWriter('Final_Bike_Data.xlsx', engine='xlsxwriter')
#
# # write each DataFrame to a specific sheet
# data_final.to_excel(writer, sheet_name='DataFinal')
#
# # Close the pandas excel writer and output the excel file
# writer.save()

# Let's look at each variable's correlation to balance
corr = data_final.corr()
cov_data = np.corrcoef(data_final.T)


# Let's visualize balance so that we do not lose the distribution once we standardize later
# plt.hist(data_final['balance'])
# plt.show()

df_desc = data_final.mean()

"""
In this section, we use our transformed data to build models that can help us predict high-value customers
"""
# We are ready to start building models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsRegressor)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
import statsmodels.api as sm


# predictors = ['bike_related_purchases', 'Age', 'tenure', 'property_valuation', 'remoteness_level',
#        'gender_Male', 'job_industry_Entertainment',
#        'job_industry_Financial Services', 'job_industry_Health',
#        'job_industry_IT', 'job_industry_Manufacturing',
#        'job_industry_Property', 'job_industry_Retail',
#        'job_industry_Telecommunications', 'wealth_segment_High Net Worth',
#        'wealth_segment_Mass Customer', 'owns_car_Yes', 'state_QLD',
#        'state_VIC']

predictors = ['tenure', 'property_valuation', 'remoteness_level', 'job_industry_Entertainment',
       'job_industry_Financial Services', 'job_industry_Health',
       'job_industry_IT', 'job_industry_Manufacturing',
       'job_industry_Property', 'job_industry_Retail',
       'job_industry_Telecommunications', 'wealth_segment_High Net Worth',
       'wealth_segment_Mass Customer', 'owns_car_Yes', 'state_QLD',
       'state_VIC']


dep_variable = ['high_value_cust']

# dummy_reg = DummyRegressor(strategy='mean')
# dummy_reg.fit(data_final[predictors], data_final[dep_variable])
#
# # This doesn't perform well, we should try normalizing our data (only continuous variable) to see if there is a better explaining power
# var_to_transform = ['bike_related_purchases', 'Age', 'tenure', 'balance', 'number_of_purchases', 'property_valuation', 'remoteness_level',
#        'gender_Male', 'job_industry_Entertainment',
#        'job_industry_Financial Services', 'job_industry_Health',
#        'job_industry_IT', 'job_industry_Manufacturing',
#        'job_industry_Property', 'job_industry_Retail',
#        'job_industry_Telecommunications', 'wealth_segment_High Net Worth',
#        'wealth_segment_Mass Customer', 'owns_car_Yes', 'state_QLD',
#        'state_VIC']
# scaler = StandardScaler()
# stand_data = pd.DataFrame(scaler.fit_transform(data_final[var_to_transform]), columns=var_to_transform)
# data_final.update(stand_data)

# We need to partition our data into train (80%) and test data (20%)
x_train, x_test, y_train, y_test = train_test_split(
   data_final[predictors],
   data_final[dep_variable],
   train_size=0.8,
   random_state=0)

# smote = SMOTE(random_state=12)
# x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
#
# y_train_res = y_train_res.values.ravel()



# Create scaler objects and train
x_scaler = StandardScaler()
# x_scaler.fit(x_train)
# y_scaler = StandardScaler()
# y_scaler.fit(y_train)

# Transform variables
# x_train = x_scaler.transform(x_train)
# x_test = x_scaler.transform(x_test)
# y_train = y_scaler.transform(y_train)
# y_test = y_scaler.transform(y_test)

x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.fit_transform(x_test)
# y_train = y_train.values.ravel()

# # Now we can start fitting models. We will start with Multiple Linear Regression
# linreg_model = LinearRegression()
# linreg_model.fit(x_train, y_train)
# print(linreg_model.score(x_train, y_train))

# y_pred = linreg_model.predict(x_test)
# y_pred = y_scaler.inverse_transform(y_pred)
# y_true = y_scaler.inverse_transform(y_test)
#
# print(mean_absolute_error(y_true, y_pred))

# # Fit model to test data
# print(linreg_model.score(x_test, y_test))


# # Let's look at each variable's correlation to balance
# corr1 = stand_data.corr()


# We didn't have good results with Linear Regression, let's see if Logistic Regression performs better
logreg_model = LogisticRegression()
logreg_model.fit(x_train, y_train)
print(logreg_model.score(x_train, y_train))
print(logreg_model.score(x_test, y_test))

y_pred = logreg_model.predict(x_test)
matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(matrix).plot()
print(classification_report(y_test, y_pred))

#
# knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(x_train, y_train)
# print(knn.score(x_train, y_train))
# print(knn.score(x_test, y_test))
#
# y_pred = knn.predict(x_test)
#
# from sklearn.model_selection import GridSearchCV
# parameters = {"n_neighbors": range(1, 10)}
# gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
# gridsearch.fit(x_train, y_train)
# gridsearch.best_params_
#
# forest = RandomForestClassifier()
# forest.fit(x_train, y_train)
# print(f'Model Accuracy: {forest.score(x_train, y_train)}')
#
#
#
# #
# disp = ConfusionMatrixDisplay(matrix).plot()
# # Fit model to test data
# print(forest.score(x_test, y_test))
#
# y_score = forest.decision_path(x_test)
# fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=forest.classes_[1])
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# #
# print(classification_report(y_test, y_pred))
