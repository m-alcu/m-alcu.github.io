---
layout: post
title: Transform category columns to boolean
date: 2017-12-10 11:35
comments: true
external-url:
categories: python pandas get_dummies
---

> Input values in Logit regression should be numerical vectors as:

* *Integer* - for example number of sales products...
* *Real* - for example millis from session user
* *Boolean* - Married or not married

Categories values are not numerical vectors, you cannot measure a value for example between married and single. This is not useful as an input for logistic regression. This type of columns should be expanded to boolean data. Pandas library has an function to do this easyly. Next example is retrieved from this [source](https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/).

Note: Before there is some sort of cleaning of some categories that could be aggregated because are too similar.

Each category column is transformed to as many columns as categories so all are now numerical vectors.

| marital |   | marital_married | marital_single | marital_divorced | marital_unknown|
|:-------:|---|:---------------:|:--------------:|:----------------:|:--------------:|
| married |   | 1               | 0              | 0                | 0              |
| single  |   | 0               | 1              | 0                | 0              |
| divorced|   | 0               | 0              | 1                | 0              |
| unknown |   | 0               | 0              | 0                | 1              |

input data can be retrieved from [here](https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv)

columns of data initial:  
['age' 'job' 'marital' 'education' 'default' 'housing' 'loan' 'contact'  
 'month' 'day_of_week' 'duration' 'campaign' 'pdays' 'previous' 'poutcome'  
 'emp_var_rate' 'cons_price_idx' 'cons_conf_idx' 'euribor3m' 'nr_employed' 'y']  

columns of data final:  
['age' 'duration' 'campaign' 'pdays' 'previous' 'emp_var_rate'  
 'cons_price_idx' 'cons_conf_idx' 'euribor3m' 'nr_employed' 'y'  

 'job_admin.'  
 'job_blue-collar'  
 'job_entrepreneur'  
 'job_housemaid'  
 'job_management'  
 'job_retired'  
 'job_self-employed'  
 'job_services'  
 'job_student'  
 'job_technician'  
 'job_unemployed'  
 'job_unknown'  

 'marital_divorced'  
 'marital_married'  
 'marital_single'  
 'marital_unknown'  

 'education_Basic'  
 'education_high.school'  
 'education_illiterate'  
 'education_professional.course'  
 'education_university.degree'  
 'education_unknown'  

 'default_no'  
 'default_unknown'  
 'default_yes'  

 'housing_no'  
 'housing_unknown'  
 'housing_yes'  

 'loan_no'  
 'loan_unknown'  
 'loan_yes'  

 'contact_cellular'  
 'contact_telephone'  

 'month_apr'  
 'month_aug'  
 'month_dec'  
 'month_jul'  
 'month_jun'  
 'month_mar'  
 'month_may'  
 'month_nov'  
 'month_oct'  
 'month_sep'  

 'day_of_week_fri'  
 'day_of_week_mon'  
 'day_of_week_thu'  
 'day_of_week_tue'  
 'day_of_week_wed'  

 'poutcome_failure'  
 'poutcome_nonexistent'  
 'poutcome_success']  

```python
import numpy as np
import pandas as pd

def main(): 

	data = pd.read_csv('banking.csv',header=0)
	data = data.dropna()
	print(data.shape)
	list(data.columns)
	data['education'].unique()

	print("reducing some enumerated values...")
	data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
	data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
	data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

	print("columns of data initial {}:").format(data.columns.values)

	print("creating dummy variables...")
	cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
	for var in cat_vars:
	    cat_list='var'+'_'+var
	    cat_list = pd.get_dummies(data[var], prefix=var)
	    data1=data.join(cat_list)
	    data=data1
	cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
	data_vars=data.columns.values.tolist()
	to_keep=[i for i in data_vars if i not in cat_vars]

	data_final=data[to_keep]
	print("columns of data final {}:").format(data_final.columns.values)

if __name__ == "__main__":
    import sys
    main()

```
