import numpy as np

if __name__ == "__main__":
	A = np.array([[1,4],[3,2]])
	B = np.array([[2,-1],[-3,4]])
	
	x = np.matmul(A,B)
	y = np.matmul(B,A)
	
	if(np.equal(x,y).all()):
		print("AB = BA")
	else:
		print("AB != BA")
		
