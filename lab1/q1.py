# assignment 1, ques 1

import numpy as np

if __name__ == "__main__":
	A = np.array([[1,2,3],[2,1,4]])
	B = np.array([[1,0],[2,1],[3,2]])
	C = np.array([[3,-1,3],[4,1,5],[2,1,3]])
	D = np.array([[2,-4,5],[0,1,4],[3,2,1]])
	E = np.array([[3,-2],[2,4]])
	
	#a
	try:
		print("A",end = " ")
		print(np.transpose(2 * A))
	except :
		print("N.A")
	
	#b
	try:
		print("B",end = " ")
		print(np.transpose(np.subtract(A,B)))
	except :
		print("N.A")
		
	#c
	try:
		print("C",end = " ")
		print(np.transpose(np.subtract(np.transpose(3*B),A)))
	except :
		print("N.A")
		
	#d
	try:
		print("D",end = " ")
		print(np.matmul(np.transpose(-A),E))
	except :
		print("N.A")
	
	#e
	try:
		print("E",end = " ")
		print(np.transpose(C + np.transpose(2*D) + E))
	except :
		print("N.A")
	
