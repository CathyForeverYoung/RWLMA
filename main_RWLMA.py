import numpy as np
import multiprocessing
from sklearn import preprocessing
from math import sqrt,log
import time
import sys
import os
import pickle

from estimator import *
from anchorselector import *
import decomposition as decomp


class Template(object):
	def __init__(self, m, n, p, dataset):
		self.dataset = dataset
		self.m = m
		self.n = n
		self.para = p

	@abc.abstractmethod
	def load_data(self):
		pass
	
	def anchor_select(self,data,TransM,q):
		# 随机游走选择锚点
		rw_anchor_selector = RWalkAnchorSelector(self.m,self.n)
		anchors = rw_anchor_selector.anchor_select(data,q,TransM)
		return anchors


	def preprocess(self,data):
		l = self.m + self.n
		#带评分的邻接矩阵
		AdjVU,TransVU = np.zeros((self.m,self.n)),np.zeros((self.m,self.n))
		for u,v,r in data:
			AdjVU[u][v] = r
		AdjUV,TransUV = AdjVU.T,np.zeros((self.n,self.m))
		#归一化得到转移矩阵
		AdjVU_sum = AdjVU.sum(axis=0)
		AdjUV_sum = AdjUV.sum(axis=0) 
		for i in range(self.n):
			if AdjVU_sum[i]>0:
				TransVU[:,i] = AdjVU[:,i]/AdjVU_sum[i]
		for i in range(self.m):
			if AdjUV_sum[i]>0:
				TransUV[:,i] = AdjUV[:,i]/AdjUV_sum[i] 
		return (TransVU,TransUV)
	
	
	def random_walk(self,TransM,anchors):
		"""不是随机开始，从某个点开始"""
		print('start random walk')
		TransVU,TransUV = TransM
		l = self.m + self.n
		q = len(anchors)
		alpha = 0.5
		#初始节点分布矩阵
		probU,probV = np.zeros((self.m, q)),np.zeros((self.n, q))
		#重启动矩阵
		restartU,restartV = np.zeros((self.m,q)),np.zeros((self.n,q))
		for i in range(q):
			au,ai = anchors[i][0],anchors[i][1]
			restartU[au][i] = 1
			probU[au][i]=1
			restartV[ai][i] = 1
		
		while True:
			probU_t = alpha*np.dot(TransVU,probV) + (1-alpha)*restartU
			probV_t = np.dot(TransUV,probU) 
			residual = np.sum(abs(probU-probU_t))+np.sum(abs(probV-probV_t))
			probU,probV = probU_t,probV_t
			if abs(residual)<1e-8:
				pU = probU.copy()
				break 

		probU[:,:],probV[:,:] = 0,0
		for i in range(q):
			au,ai = anchors[i][0],anchors[i][1]
			probV[ai][i]=1
		while True:
			probV_t = alpha*np.dot(TransUV,probU) + (1-alpha)*restartV 
			probU_t = np.dot(TransVU,probV) 
			residual = np.sum(abs(probU-probU_t))+np.sum(abs(probV-probV_t))
			probU,probV = probU_t,probV_t
			if abs(residual)<1e-8:
				pV = probV.copy()
				break 

		return (pU,pV)
	

	def submatrix_const(self,prob,q,data_train,data_test):
		print('start constructing submatrices')
		#分别得到用户物品的稳态概率矩阵
		probU,probV = prob
		anchor_neighuser = {}
		anchor_neighitem = {}
		for u in range(self.m):
			index_val = [(i,j) for i,j in enumerate(probU[u])]
			index_val = sorted(index_val,key=lambda s: s[1],reverse=True)[:int(q*self.para)]
			for p in index_val:
				anchor_neighuser.setdefault(p[0],[])
				anchor_neighuser[p[0]].append(u)
		for v in range(self.n):
			index_val = [(i,j) for i,j in enumerate(probV[v])]
			index_val = sorted(index_val,key=lambda s: s[1],reverse=True)[:int(q*self.para)]
			for p in index_val:
				anchor_neighitem.setdefault(p[0],[])
				anchor_neighitem[p[0]].append(v)

		subdata_train,subdata_test = {},{}
		nargs = [(data_train,data_test,anchor_neighuser[i],anchor_neighitem[i],i) for i in range(q) if (i in anchor_neighuser.keys()) and (i in anchor_neighitem.keys())]
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 2)
		for y in pool.imap(self.get_subdata,nargs):
			this_train, this_test, i = y
			subdata_train[i] = this_train
			subdata_test[i] = this_test
		pool.close()
		pool.join()

		return (subdata_train,subdata_test)	


	def local_train(self,q,anchors,subdata_train,subdata_test):
		print('start local train')
		multiprocessing.freeze_support()
		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores - 2)
		
		eachpred_dict = {}	
		nargs = [(subdata_train[i], subdata_test[i], q, i) for i in range(q) if i in subdata_train.keys()]
		for y in pool.imap(self.train_submatrix, nargs):
			pred_rate, subdata_test_i, i = y
			self.fill_pred_dict(eachpred_dict, pred_rate, subdata_test_i, q, i)
			sys.stdout.write('have finished training for %d/%d local matrices\r' % (i+1,q))
		pool.close()
		pool.join()
		return eachpred_dict


	def predict(self,data_test,eachpred_dict):
		#子矩阵中测试数据评分
		true_dict_test = self.get_datadic(data_test)
		pred_dict = {}
		for user in eachpred_dict:
			pred_dict.setdefault(user, {})
			for item in eachpred_dict[user]:
				ratesum = ratenum = 0
				for i in eachpred_dict[user][item]:
					if i != 0:
						ratesum += i
						ratenum += 1
				pred_dict[user][item] = ratesum / ratenum
		
		estimator = Estimator()
		mae = round(estimator.get_mae(pred_dict, true_dict_test),4)
		rmse = round(estimator.get_rmse(pred_dict, true_dict_test),4)
		print("---------directly average result is, MAE:" + str(mae) + ";RMSE:" + str(rmse)+'----------')
		return mae,rmse


	@abc.abstractmethod
	def result_print(self):
		pass


	def get_subdata(self,args):
		"""
		工具方法：找到子矩阵的点
		"""
		data_train,data_test,neighuser,neighitem,i = args
		subdata_train,subdata_test = [],[]
		for d in data_train:
			if (d[0] in neighuser) and (d[1] in neighitem):
				subdata_train.append(d)
		for d in data_test:
			if (d[0] in neighuser) and (d[1] in neighitem):
				subdata_test.append(d)
		return subdata_train,subdata_test,i


	def train_submatrix(self,args):
		"""
		工具方法：服务于并行训练
		"""
		data_train,data_test,q,i = args
		svd = decomp.SVD(data_train, k=10)
		svd.train(data_test, steps=45, gamma=0.01, Lambda=0.001)
		pred_rate = svd.test(data_test)
		return (pred_rate, data_test, i)


	def fill_pred_dict(self,dict_data,pred,test,len_q,q):
		"""
		工具方法：将预测的值填入字典
		"""
		for i in range(len(test)):
			dict_data.setdefault(test[i][0],{})
			dict_data[test[i][0]].setdefault(test[i][1],np.zeros(len_q))
			dict_data[test[i][0]][test[i][1]][q]=pred[i]


	def get_datadic(self,data):
		"""
		工具方法：构建满足需求的字典 
		"""
		true_dict={}
		for i in range(len(data)):
			uid = data[i][0]
			mid = data[i][1]
			rate = data[i][2]
			true_dict.setdefault(uid, {})
			true_dict[uid][mid] = rate
		return true_dict


class ML100k(Template):
	def load_data(self,fold=1):
		print("load data")
		data_train,data_test = [],[]
		file_path_train = "ml-100k/u"+str(fold)+".base"
		file_path_test = "ml-100k/u"+str(fold)+".test"
		with open(file_path_train) as f:
			for line in f:
				a = line.split("\t")
				data_train.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		with open(file_path_test) as f:
			for line in f:
				a = line.split("\t")
				data_test.append((int(a[0]) - 1, int(a[1]) - 1, int(a[2])))  # 此处已经把id变成索引（-1）
		return data_train,data_test


class ML1m(Template):
	def load_data(self,fold=1):
		with open("ml-1m/data/train"+str(fold)+".txt",'r') as f:
			data_train = eval(f.read())
		with open("ml-1m/data/test"+str(fold)+".txt",'r') as f:
			data_test = eval(f.read())
		return (data_train,data_test)


class Ciao(Template):
	def load_data(self,fold=1):
		with open("ciao/data/train"+str(fold)+".txt",'r') as f:
			data_train = eval(f.read())
		with open("ciao/data/test"+str(fold)+".txt",'r') as f:
			data_test = eval(f.read())
		return (data_train,data_test)
		

if __name__=='__main__':
	p = 0.7
	q = 50

	# DataInstance = ML100k(m=943,n=1682,p=p, dataset = "ML100K")
	# DataInstance = ML1m(m=6040,n=3952,p=p, dataset = "ML1M")
	DataInstance = Ciao(m=7375,n=106797,p=p, dataset = "Ciao")
	

	for fold in range(1,2):
		print("------the dataset running is {}; the fold is {}-------".format(DataInstance.dataset,fold))
		
		#加载数据
		data_train,data_test = DataInstance.load_data(fold)
		#特征提取,得到状态转移矩阵
		TransM = DataInstance.preprocess(data_train)
		#选择锚点
		anchors = DataInstance.anchor_select(data_train,TransM,q)	
		#随机游走寻找邻域
		anchorM = DataInstance.random_walk(TransM,anchors)
		#得到以每个锚点为中心子矩阵所包含的训练集和测试集
		subdata_train,subdata_test = DataInstance.submatrix_const(anchorM,q,data_train,data_test)
		
		#训练
		eachpred_dict = DataInstance.local_train(q,anchors,subdata_train,subdata_test)		
		#预测
		mae1,rmse1 = DataInstance.predict(data_test,eachpred_dict)
	
		a=0
		for i in subdata_train:		
			a = a + len(subdata_train[i])
		print('平均子矩阵大小',a/len(subdata_train))		