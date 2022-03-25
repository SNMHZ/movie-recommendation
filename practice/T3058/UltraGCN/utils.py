import numpy as np
import pickle
import torch
'''
Useful functions
'''

def pload(path):
	with open(path, 'rb') as f: #binary형식 읽기
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f: #binary형식 쓰기
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


