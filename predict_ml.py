# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

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
data['suicide'][data['suicide']==1]=2
data['suicide'][data['suicide']==0]=1
data['suicide'][data['suicide']==2]=0

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

data_xx_arr=np.array(data_xx)
data_y_arr=np.array(data_y3)
data_xx, data_y = RandomUnderSampler(random_state=1234).fit_sample(data_xx_arr, data_y_arr)
kf = StratifiedKFold(n_splits=5,random_state=18, shuffle=False)

set_number=0
data_result=pd.DataFrame(columns=['parameter','mean_result'])
for train_index, test_index in kf.split(data_xx,data_y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    x_train, x_test = data_xx[train_index], data_xx[test_index]
    y_train, y_test = data_y[train_index], data_y[test_index]

    para=pd.read_csv("./csv/LR_%s_parameter.csv"%str(set_number))
    para=para.sort_values(by='mean_result',ascending=False)
    para=para.reset_index()
    
    dic={para['parameter'][0].replace("{","").replace("}","").split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(":")[1].strip()}
    LR=LogisticRegression(C=float(dic['C']))    
    LR.fit(x_train,y_train)
    sdf_LR=LR.decision_function(data_xx_arr)
    
    np.save("./result/all_%s_true.npy"%str(set_number),data_y_arr)    
    np.save("./result/LR_all_%s_pred.npy"%str(set_number),sdf_LR)

    para=pd.read_csv("./csv/SVM_%s_parameter.csv"%str(set_number))
    para=para.sort_values(by='mean_result',ascending=False)
    para=para.reset_index()
    
    dic={para['parameter'][0].replace("{","").replace("}","").split(",")[0].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[0].split(":")[1].strip(),
         para['parameter'][0].replace("{","").replace("}","").split(",")[1].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[1].split(":")[1].strip(),
         para['parameter'][0].replace("{","").replace("}","").split(",")[2].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[2].split(":")[1].strip()
         }
    svc=SVC(C=float(dic['C']),gamma=float(dic['gamma']), kernel='rbf')
    
    svc.fit(x_train,y_train)
    sdf_SVC=svc.decision_function(data_xx_arr)
    np.save("./result/SVM_all_%s_pred.npy"%str(set_number),sdf_SVC)

    para=pd.read_csv("./csv/RF_%s_parameter.csv"%str(set_number))
    para=para.sort_values(by='mean_result',ascending=False)
    para=para.reset_index()
    
    dic={para['parameter'][0].replace("{","").replace("}","").split(",")[0].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[0].split(":")[1].strip(),
         para['parameter'][0].replace("{","").replace("}","").split(",")[1].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[1].split(":")[1].strip(),
         para['parameter'][0].replace("{","").replace("}","").split(",")[2].split(":")[0].replace("'","").strip():para['parameter'][0].replace("{","").replace("}","").split(",")[2].split(":")[1].strip()
         }
    rf=RandomForestClassifier(max_depth=int(dic['max_depth']),min_samples_split=int(dic['min_samples_split']),min_samples_leaf=int(dic['min_samples_leaf']))

    rf.fit(x_train,y_train)
    sdf_rf=rf.predict_proba(data_xx_arr)[:,1]
    np.save("./result/RF_all_%s_pred.npy"%str(set_number),sdf_rf)
    
    set_number=set_number+1
