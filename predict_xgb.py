# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,auc

data=pd.read_csv("./Data/alldata.csv",encoding="MS949",engine="python")
data=data.drop(['BMI'],axis=1)
data=data.drop(['INT_problem'],axis=1)

drop_list=[]
for i in range(len(data)):
    if data['OBE'][i]==" ":
        drop_list.append(i)
data=data.drop(drop_list,0)
data=data.reset_index(drop=True)
data['OBE']=data['OBE'].astype('int64')

drop_list=[]
for i in range(len(data)):
    if data['EDU_F'][i]==" ":
        drop_list.append(i)
data=data.drop(drop_list,0)
data=data.reset_index(drop=True)
data['EDU_F']=data['EDU_F'].astype('int64')

drop_list=[]
for i in range(len(data)):
    if data['D_WT_P'][i]==" ":
        drop_list.append(i)
data=data.drop(drop_list,0)
data=data.reset_index(drop=True)
data['D_WT_P']=data['D_WT_P'].astype('int64')

all_col_name=list(data.columns)
data_y1=data['SI']
data_y2=data['SA']
data_y3=data['suicide']

data_x=data.copy()
y_col_list=['SI','SA','suicide']
for y_col in y_col_list:
    data_x=data_x.ix[:, data_x.columns != y_col ]

data_x['MH_SCHOOL'][data_x['MH_SCHOOL']==1]=0
data_x['MH_SCHOOL'][data_x['MH_SCHOOL']==2]=1

data_x['SEX2'][data_x['SEX2']==1]=0
data_x['SEX2'][data_x['SEX2']==2]=1

data_xx=data_x[['OBE','CA','CS','SEX2','SAD','SUBS','MH_SCHOOL','PA','INT_A','D_WT_P','ASTHMA','AR','AD','SE','SCH_I','VIO']]
x_col_list=data_x.columns
for x_col in x_col_list:
    if not x_col in ['OBE','CA','CS','SEX2','SAD','SUBS','MH_SCHOOL','PA','INT_A','D_WT_P','ASTHMA','AR','AD','SE','SCH_I','VIO']:
        data_xx=pd.concat([data_xx,pd.get_dummies(data_x[x_col], prefix=x_col)],axis=1)
x_col_list=data_xx.columns

data_xx=np.array(data_xx)
data_y=np.array(data_y3)

kf = StratifiedKFold(n_splits=10,random_state=18, shuffle=False)
kf2 = StratifiedKFold(n_splits=5,random_state=28, shuffle=False)

set_number=0

for train_index, test_index in kf.split(data_xx,data_y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    x_train, x_test = data_xx[train_index], data_xx[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]
    set_number_2=0
    result_xgb_list=[]
    for train_index_2, test_index_2 in kf2.split(x_train,y_train):
        print("TRAIN:", len(train_index_2), "TEST:", len(test_index_2))
        Xtrain, Xtest = x_train[train_index_2], x_train[test_index_2]
        ytrain, ytest = y_train[train_index_2], y_train[test_index_2]
        
        para_xgb=pd.read_csv("./csv/%s_parameters_%s.csv"%(str(set_number),str(set_number_2)))
        para_xgb=para_xgb.sort_values(by='8AUC',ascending=False)
        para_xgb=para_xgb.reset_index()
        
        lr=para_xgb['2lr'][0]
        colsample=para_xgb['3colsample'][0]
        max_depth=para_xgb['4max_depth'][0]
        gamma=para_xgb['5gamma'][0]
        reg_l2=para_xgb['6reg_l2'][0]
        reg_l1=para_xgb['7reg_l1'][0]
        
        params={"learning_rate":lr,"n_estimators":5000,"max_depth":max_depth,"colsample_bytree":colsample,"gamma":gamma,"lamdba":reg_l2,"alpha":reg_l1,"tree_method":"gpu_exact",
                "predictor":"gpu_predictor","objective":"binary:logistic"}
        clf=xgb.XGBClassifier(**params)
        clf.fit(Xtrain,ytrain,eval_set=[(Xtrain,ytrain),(Xtest,ytest)],eval_metric='auc',early_stopping_rounds=100,verbose=50)
        result_xgb=clf.predict_proba(Xtest)[:,1]
        
        fp_xgb,tp_xgb,_=roc_curve(ytest,result_xgb)
        auc_xgb=auc(fp_xgb, tp_xgb)
        result_xgb_list.append(auc_xgb)
        set_number_2=set_number_2+1
    max_index=result_xgb_list.index(max(result_xgb_list))
    
    set_number_2=0
    for train_index_2, test_index_2 in kf2.split(x_train,y_train):
        print("TRAIN:", len(train_index_2), "TEST:", len(test_index_2))
        Xtrain, Xtest = x_train[train_index_2], x_train[test_index_2]
        ytrain, ytest = y_train[train_index_2], y_train[test_index_2]
        if set_number_2==max_index:
        
            para_xgb=pd.read_csv("./csv/%s_parameters_%s.csv"%(str(set_number),str(set_number_2)))
            para_xgb=para_xgb.sort_values(by='8AUC',ascending=False)
            para_xgb=para_xgb.reset_index()
            
            lr=para_xgb['2lr'][0]
            colsample=para_xgb['3colsample'][0]
            max_depth=para_xgb['4max_depth'][0]
            gamma=para_xgb['5gamma'][0]
            reg_l2=para_xgb['6reg_l2'][0]
            reg_l1=para_xgb['7reg_l1'][0]
            
            params={"learning_rate":lr,"n_estimators":5000,"max_depth":max_depth,"colsample_bytree":colsample,"gamma":gamma,"lamdba":reg_l2,"alpha":reg_l1,"tree_method":"gpu_exact",
                    "predictor":"gpu_predictor","objective":"binary:logistic"}
            clf=xgb.XGBClassifier(**params)
            clf.fit(Xtrain,ytrain,eval_set=[(Xtrain,ytrain),(Xtest,ytest)],eval_metric='auc',early_stopping_rounds=100,verbose=50)
            result_xgb=clf.predict_proba(x_test)[:,1]
        
            np.save("./result/%s_xgb.npy"%(str(set_number)),result_xgb)
            set_number_2=set_number_2+1
        
    set_number=set_number+1
