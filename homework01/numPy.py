import numpy as np

mu, sigma = 0, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, (5,5))

print(s)

for i in range(5):
	for j in range(5):
		print(s[i,j])
		if (s[i,j] > 0.09):
			print("true")
			s[i,j] = s[i,j]**2
		else :
			s[i,j] = 42

print(s)
