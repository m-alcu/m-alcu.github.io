---
layout: post
title: Transform enumerated data
date: 2017-12-10 11:35
comments: true
external-url:
categories: python pandas get_dummies
---

> Input values in Logit regression should be numerical vectors as:

* Integer - for example number of sales products...
* Real - for example millis from session user
* Boolean - Married or not married

Categories values should be expanded to boolean data. Pandas library has an function to do this easyly. Next example is retrieved from this source [source](https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/).

Note: Before there is some sort of cleaning of some categories that could be aggregated because are too similar.

Each category column is transformed to as many columns as categories so all are now numerical vectors.

| marital |   | marital_married | marital_single |
|:-------:|---|:---------------:|:--------------:|
| married |   | 1               | 0              |
| single  |   | 0               | 1              |


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
