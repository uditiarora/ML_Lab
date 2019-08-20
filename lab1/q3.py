import numpy as np
import math 
def isOrthogonal(vectors):
	for i in range(len(vectors)):
		for j in range(i+1,len(vectors)):
			if(vectors[i][0]*vectors[j][0] + vectors[i][1]*vectors[j][1] + vectors[i][2]*vectors[j][2] == 0):
				return True
	return False

def makeOrthonormal(vectors):
	if(isOrthogonal(vectors)):
		for i in range(len(vectors)):
			temp = (vectors[i][0]**2 + vectors[i][1]**2 + vectors[i][2]** 2)**0.5
			vectors[i][0] = vectors[i][0] / temp
			vectors[i][1] = vectors[i][1] / temp
			vectors[i][2] = vectors[i][2] / temp
	return vectors
if __name__ == "__main__":
	v1 = [-2,0,10]
	v2 = [0,1,0]
	v3 = [2,0,4]
	vectors = (v1,v2,v3)
	print(isOrthogonal(vectors))
	
	print(makeOrthonormal(vectors))
