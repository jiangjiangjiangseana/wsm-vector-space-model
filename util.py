import sys
from scipy.spatial import distance

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
	import numpy as np
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclidean(vector1, vector2):
	# np_v1 = np.array(vector1)
	# np_v2 = np.array(vector2)
	# return float(np.sqrt(np.sum(np.square(np_v1 - np_v2)))) 
	# return float(np.linalg.norm(np_v1 - np_v2))
	# return float(norm(np_v1-np_v2))
	return float(distance.euclidean(vector1, vector2))
