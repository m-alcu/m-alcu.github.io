---
layout: post
title: Recursive Feature Elimination (RFE)
date: 2017-12-10 18:30
comments: true
external-url:
categories: python sklearn feature_selection
---

> In case there are to many features it is reasonable, for optimization reasons, to select the best performance features. [Recursive Feature Elimination (RFE)](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) does the job to get recursively smaller groups of features giving a `_coef` on each feature to reach to desired number of features.

The process needs 

input data can be retrieved from [here](https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv)


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

	data_final_vars=data_final.columns.values.tolist()
	y=['y']
	X=[i for i in data_final_vars if i not in y]

	print("18 best feature selection...")
	logreg = LogisticRegression()
	rfe = RFE(logreg, 18)
	rfe = rfe.fit(data_final[X], data_final[y] )
	print(rfe.support_)
	print(rfe.ranking_)

	print("The Recursive Feature Elimination (RFE) has helped us select the following features:")

	#Iterate from array de rfe.support_ and pick columns that are == True
	i = 0
	cols = []
	for e in rfe.support_:
	    if e == True:
	        cols.append(X[i])
	    i = i+1

	X=data_final[cols]
	y=data_final['y']
	

if __name__ == "__main__":
    import sys
    main()

```
