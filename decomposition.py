import numpy as np
from collections import defaultdict

class  SVD:
    def __init__(self,X,k=20):
        '''''
            k  is the length of vector
        '''
        self.X=np.array(X)
        self.k=k
        self.ave=np.mean(self.X[:,2])
        #print("the train data size is ",self.X.shape)
        self.bi={}
        self.bu={}
        self.qi={}
        self.pu={}
        
        for i in range(self.X.shape[0]):
            uid=self.X[i][0]
            mid=self.X[i][1]
            rat=self.X[i][2]
            self.bi.setdefault(mid,0)
            self.bu.setdefault(uid,0)
            self.qi.setdefault(mid,np.random.random((self.k,1))/10*(np.sqrt(self.k)))
            self.pu.setdefault(uid,np.random.random((self.k,1))/10*(np.sqrt(self.k)))

    def pred(self,uid,mid):
        self.bi.setdefault(mid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(mid,np.zeros((self.k,1)))
        self.pu.setdefault(uid,np.zeros((self.k,1)))
        if (self.qi[mid] is None):
            self.qi[mid]=np.zeros((self.k,1))
        if (self.pu[uid] is None):
            self.pu[uid]=np.zeros((self.k,1))
        ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])
        if ans>5:
            return 5
        elif ans<1:
            return 1
        return ans

    def train(self,data_test,steps=20,gamma=0.04,Lambda=0.15,changerate=0.975):
        for step in range(steps):
            #print('the ',step,'th  step is running')
            rmse_sum=0.0
            kk=np.random.permutation(self.X.shape[0])    #np.random.permutation(5)  [3,1,2,0,4] 顺序打乱
            for j in range(self.X.shape[0]):
                i=kk[j]
                uid=self.X[i][0]
                mid=self.X[i][1]
                rat=self.X[i][2]
                # k1=usim[uid][q]
                # k2=isim[mid][q]
                #参数更新
                eui=rat-self.pred(uid,mid)
                rmse_sum+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])
                temp=self.qi[mid]
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])

            #学习速率递减
            gamma=gamma*changerate
        return 

    def test(self,test_X):
        output=[]
        sums=0
        test_X=np.array(test_X)
        #print("the test data size is ",test_X.shape)
        for i in range(test_X.shape[0]):
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)
            sums+=(pre-test_X[i][2])**2
        rmse=np.sqrt(sums/test_X.shape[0])
        #print("-------------------------------the rmse on test data is",rmse)
        return output


