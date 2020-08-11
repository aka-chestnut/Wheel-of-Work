from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import time
from pprint import pprint
class XGBPipeline():
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)        
        self.X_test = np.array(X_test)     
        self.y_test = np.array(y_test)
    
    def get_ks(y_true,y_pred):
        return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    
    def timer (func):
        def wrapper(*args,**kwargs): 
            start = time.time()
            result = func(*args,**kwargs)
            end = time.time()
            print(func.__name__+'运行时间：','{:.2f}'.format(end-start))
            return result
        return wrapper
    
    @timer
    def fit_baseline(self):
        print("==========Model BaseLine Train==========")
        params={'booster':'gbtree',
                'objective': 'binary:logistic',
                'max_depth':5,
                'subsample':0.8,
                'colsample_bytree':0.8,
                'min_child_weight':8,
                'learning_rate ': 0.0005,
                'nthread':-1,
                'n_estimators':1000}
        model = XGBClassifier(**params,verbosity=2)
        model.fit(self.X_train,self.y_train,verbose=2)
        self.model_evl(model)
        
    @timer
    def train_model(self,params):
        model = XGBClassifier(**params,verbosity=2)
        model.fit(self.X_train,self.y_train,verbose=2)
        self.model_evl(model)
        return model
        
    @timer
    def gridsearch_para(self):
        params={'booster':'gbtree',
                'objective': 'binary:logistic',
                'max_depth':5,
                'subsample':0.8,
                'colsample_bytree':0.8,
                'min_child_weight':8,
                'learning_rate ': 0.0005,
                'nthread':-1,
                'n_estimators':1000}
        print("==========Optimize Parameters==========")
        print("n_estimators:")
        param_test1 = {'n_estimators':range(150,300,10)}
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            seed=2018,
            silent=False)
        gsearch1 = GridSearchCV(estimator = xgb, param_grid = param_test1, scoring='roc_auc',cv=5,n_jobs=-1)
        gsearch1.fit(self.X_train,self.y_train)
        print(gsearch1.best_params_,gsearch1.best_score_)
        params.update(gsearch1.best_params_)
        print("max_depth:")
        param_test2 = {'max_depth':range(3,8,1)}
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            seed=2018,
            silent=False)
        gsearch2 = GridSearchCV(estimator = xgb, param_grid = param_test2, scoring='roc_auc',cv=5,n_jobs=-1)
        gsearch2.fit(self.X_train,self.y_train)
        print(gsearch2.best_params_,gsearch2.best_score_)
        params.update(gsearch2.best_params_)
        print("min_child_weight:")
        param_test3 = {'min_child_weight':range(1,10,1)}
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            seed=2018,
            silent=False)
        gsearch3 = GridSearchCV(estimator = xgb, param_grid = param_test3, scoring='roc_auc',cv=5,n_jobs=-1)
        gsearch3.fit(self.X_train,self.y_train)
        print(gsearch3.best_params_,gsearch3.best_score_)
        params.update(gsearch3.best_params_)
        print("subsample:")
        param_test4 = {
         'subsample':[i/10.0 for i in range(6,10,1)],
        #  'colsample_bytree':[i/10.0 for i in range(6,10,1)]
        }
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            seed=2018,
            silent=False)
        gsearch4 = GridSearchCV(estimator = xgb, param_grid = param_test4, scoring='roc_auc',cv=5,n_jobs=-1)
        gsearch4.fit(self.X_train,self.y_train)
        print(gsearch4.best_params_,gsearch4.best_score_)
        params.update(gsearch4.best_params_)
        print("colsample_bytree:")
        param_test5 = {
          'colsample_bytree':[i/10.0 for i in range(6,10,1)]
        }
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            seed=2018,
            silent=False)
        gsearch5 = GridSearchCV(estimator = xgb, param_grid = param_test5, scoring='roc_auc',cv=5,n_jobs=-1)
        gsearch5.fit(self.X_train,self.y_train)
        print(gsearch5.best_params_,gsearch5.best_score_)
        params.update(gsearch5.best_params_)
        print("调参结束后模型效果:")
        params.update({'scale_pos_weight':float(len(self.y_train)-np.sum(self.y_train))/float(np.sum(self.y_train)),
            'seed':2018,
            'silent':False})
        self.train_model(params)
        
    def model_evl(self,model):
        print('模型评价：')
        pred = model.predict_proba(self.X_test)
        pred_y = pred[:,1]
        print ('AUC: %.4f' % metrics.roc_auc_score(self.y_test,pred_y))
        ypred = (pred_y>=0.5)*1 
        print ('ACC: %.4f' % metrics.accuracy_score(self.y_test,ypred))
        print ('Recall: %.4f' % metrics.recall_score(self.y_test,ypred))
        print ('Precesion: %.4f' %metrics.precision_score(self.y_test,ypred))
        print ('F1-score: %.4f' %metrics.f1_score(self.y_test,ypred))
        print ('KS: %.4f' %get_ks(self.y_test,ypred))
        print('\n')
        print(metrics.confusion_matrix(self.y_test,ypred)) 
        