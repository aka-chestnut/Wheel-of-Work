'''
Author:huoyk<huoyongkang@outlook.com
Created Time:2020年8月13日 10:12:48
Description:A Method Set For Binary Classification Problem
'''
import time
import seaborn as sns
import pandas as pd
class PUClassificationEDA(object):
    def __init__(self,df,target_name):
        self.df = df
        self.target_name = target_name
    
    def __str__(self):
        length = len(self.df)
        feature_num = df.shape[1]
        ratio0 = len(self.df[self.df[self.target_name]==0]) / len(self.df)
        ratio1 = len(self.df[self.df[self.target_name]==1]) / len(self.df)
        return '<lenth:%.0f,feature_num:%.0f,ratio0:%.6f,ratio1:%.6f>' %(length,feature_num,ratio0,ratio1)
    
    def timer (func):
        def wrapper(*args,**kwargs): 
            start = time.time()
            result = func(*args,**kwargs)
            end = time.time()
            print(func.__name__+' 运行时间：','{:.2f} min'.format((end-start)/60))
            return result
        return wrapper
    
    @timer
    def get_null_dist_difference(self):
        nulldf = round(self.df.isnull().sum() / len(self.df),6)
        PositiveNull = round(self.df[self.df[self.target_name]==1].isnull().sum() / len(self.df[self.df[self.target_name]==1]),6)
        UnlabelNull = round(self.df[self.df[self.target_name]==0].isnull().sum() / len(self.df[self.df[self.target_name]==0]),6)
        nulldf = pd.DataFrame(nulldf,columns=['total'])
        nulldf['UnlabelNull'] = UnlabelNull.tolist()
        nulldf['PositiveNull'] = PositiveNull.tolist()
        nulldf['UminusP'] = round(nulldf['UnlabelNull'] - nulldf['PositiveNull'],6)
        return nulldf

    @timer
    def get_feature_dist_difference(self,col_name):
        df0 = self.df[self.df[self.target_name]==0].groupby(col_name)[self.target_name].count().sort_values(ascending=False) / len(self.df[self.df[self.target_name]==0])
        df1 = self.df[self.df[self.target_name]==1].groupby(col_name)[self.target_name].count().sort_values(ascending=False) / len(self.df[self.df[self.target_name]==1])
        df0 = pd.DataFrame(df0)
        df0.columns = ['ratio0']
        df0 = df0.join(df1,how='outer')
        df0.columns = ['ratio0','ratio1']
        df0['1minus0'] = df0['ratio1'] - df0['ratio0']
        df0.sort_values(by='1minus0',ascending=False,inplace=True)
        return df0
    
    @timer
    def get_feature_dtype(self):
        object_columns_df = df.select_dtypes(include=['object'])
        numerical_columns_df =df.select_dtypes(exclude=['object'])
        return object_columns_df.columns.tolist(),numerical_columns_df.columns.tolist()
    
    @timer
    def plot_category_heatmap(self,target=0,offticks=True):
        corrmat = self.df[self.df['target']==target].corr()
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)
        if offticks==True:
            plt.xticks([])  
            plt.yticks([])
        return corrmat
