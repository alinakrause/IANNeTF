
limit = 3 

def generator(limit):
     
	list = range(limit)
	for i in list:
		yield i

		


generator = generator(limit)
for i in generator:
	for j in generator:
		print("Meow ")

