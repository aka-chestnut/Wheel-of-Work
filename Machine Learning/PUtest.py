# Author:huoyk
# Created Date:2020年8月13日 18:20:39
# Description:PU问题的两个跨期样本测试，测试结果自动保存为npy，定义PU专用指标，生成并保存
from pprint import pprint
import numpy as np
import time
class PUtest(object):
    def __init__(self,model,X_test1,y_test1,X_test2,y_test2,model_name):
        self.model = model
        self.X_test1 = X_test1
        self.y_test1 = y_test1
        self.X_test2 = X_test2
        self.y_test2 = y_test2
        self.model_name = model_name
        
    def get_ks(self,y_true,y_pred):
        return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    
    def timer (func):
        def wrapper(*args,**kwargs): 
            start = time.time()
            result = func(*args,**kwargs)
            end = time.time()
            print(func.__name__+' 运行时间：','{:.2f分钟}'.format((end-start) / 60))
            return result
        return wrapper 
    
    @timer
    def traditional_model_evl(self):
        dict = {}
        print('==========跨期测试集1==========：')
        try:
            pred = self.model.predict_proba(self.X_test1)
            pred_y = pred[:,1]
        except:
            pred = self.model.predict(self.X_test1)
            pred_y = pred
        dict_part1 = {}
        dict_part1['AUC'] = metrics.roc_auc_score(self.y_test1,pred_y)
        ypred = (pred_y>=0.5)*1 
        dict_part1['ACC'] = metrics.accuracy_score(self.y_test1,ypred)
        dict_part1['Recall'] = metrics.recall_score(self.y_test1,ypred)
        dict_part1['Precesion'] = metrics.precision_score(self.y_test1,ypred)
        dict_part1['F1-score'] = metrics.f1_score(self.y_test1,ypred)
        dict_part1['KS'] = self.get_ks(self.y_test1,ypred)
        dict_part1['confusion_matrix'] = metrics.confusion_matrix(self.y_test1,ypred)
        print ('AUC: %.4f' % dict_part1['AUC'])
        print ('ACC: %.4f' % dict_part1['ACC'])
        print ('Recall: %.4f' % dict_part1['Recall'])
        print ('Precesion: %.4f' % dict_part1['Precesion'])
        print ('F1-score: %.4f' % dict_part1['F1-score'])
        print ('KS: %.4f' % dict_part1['KS'])
        print('\n')
        pprint(dict_part1['confusion_matrix'])  
        dict['val1'] = dict_part1
        print('==========跨期测试集2==========：')
        try:
            pred = self.model.predict_proba(self.X_test2)
            pred_y = pred[:,1]
        except:
            pred = self.model.predict(self.X_test2)
            pred_y = pred
        dict_part2 = {}
        dict_part2['AUC'] = metrics.roc_auc_score(self.y_test2,pred_y)
        ypred = (pred_y>=0.5)*1 
        dict_part2['ACC'] = metrics.accuracy_score(self.y_test2,ypred)
        dict_part2['Recall'] = metrics.recall_score(self.y_test2,ypred)
        dict_part2['Precesion'] = metrics.precision_score(self.y_test2,ypred)
        dict_part2['F1-score'] = metrics.f1_score(self.y_test2,ypred)
        dict_part2['KS'] = self.get_ks(self.y_test2,ypred)
        dict_part2['confusion_matrix'] = metrics.confusion_matrix(self.y_test2,ypred)
        print ('AUC: %.4f' % dict_part2['AUC'])
        ypred = (pred_y>=0.5)*1 
        print ('ACC: %.4f' % dict_part2['ACC'])
        print ('Recall: %.4f' % dict_part2['Recall'])
        print ('Precesion: %.4f' % dict_part2['Precesion'])
        print ('F1-score: %.4f' % dict_part2['F1-score'])
        print ('KS: %.4f' % dict_part2['KS'])
        print('\n')
        pprint(dict_part2['confusion_matrix'])  
        dict['val2'] = dict_part2
        np.save('model_result/'+self.model_name+'.npy',dict)
                  
    def pu_recall(self,true,pred):
        return np.sum(pred[true==1]) / np.sum(true)
                  
    def pu_precesion(self,true,pred):
        return np.sum(true[pred==1]) / np.sum(pred)
                  
    def pu_f1(self,true,pred):
        recall = pu_recall(true,pred)
        precesion = pu_precesion(true,pred)
        return 2*recall*precesion / (recall+precesion)

    def pu_adjusted_f1(self,true,pred):
        tmp1 = sum(pred) / len(pred)
        tmp2 = np.sum(pred[true==1]) / np.sum(true)
        return tmp2 * tmp2 / tmp1
                  
    @timer
    def pu_model_evl(self,model):
        # 重新定义模型评价指标，适用于PU问题
        dict = {}
        print('==========跨期测试集1==========：')
        try:
            pred = self.model.predict_proba(self.X_test1)
            pred_y = pred[:,1]
        except:
            pred = self.model.predict(self.X_test1)
            pred_y = pred
        pred_y = pred[:,1]
        ypred = (pred_y>=0.5)*1 
        dict_part1 = {}
        dict_part1['PU_Recall'] = self.pu_recall(self.y_test1,ypred)
        dict_part1['PU_Precesion'] = self.pu_precesion(self.y_test1,ypred)
        dict_part1['PU_F1'] = self.pu_f1(self.y_test1,ypred)
        dict_part1['PU_Adjusted_F1'] = self.pu_adjusted_f1(self.y_test1,ypred)
        print ('PU_Recall: %.4f' % dict_part1['PU_Recall'])
        print ('PU_Precesion: %.4f' % dict_part1['PU_Precesion'])                  
        print ('PU_F1: %.4f' % dict_part1['PU_F1'])                  
        print ('PU_Adjusted_F1: %.4f' % dict_part1['PU_Adjusted_F1'])
        dict['val1'] = dict_part1
        print('==========跨期测试集2==========：')
        try:
            pred = self.model.predict_proba(self.X_test2)
            pred_y = pred[:,1]
        except:
            pred = self.model.predict(self.X_test2)
            pred_y = pred
        ypred = (pred_y>=0.5)*1 
        dict_part2 = {}
        dict_part2['PU_Recall'] = self.pu_recall(self.y_test2,ypred)
        dict_part2['PU_Precesion'] = self.pu_precesion(self.y_test2,ypred)
        dict_part2['PU_F1'] = self.pu_f1(self.y_test2,ypred)
        dict_part2['PU_Adjusted_F1'] = self.pu_adjusted_f1(self.y_test2,ypred)
        print ('PU_Recall: %.4f' % dict_part2['PU_Recall'])
        print ('PU_Precesion: %.4f' % dict_part2['PU_Precesion'])                  
        print ('PU_F1: %.4f' % dict_part2['PU_F1'])                  
        print ('PU_Adjusted_F1: %.4f' % dict_part2['PU_Adjusted_F1'])
        dict['val2'] = dict_part2
        np.save('model_result_PU/'+self.model_name+'.npy',dict)