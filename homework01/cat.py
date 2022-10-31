

class Cat:
	pass

	def __init__(self, name):
		self.name = name

 
	def greet(self,other_cat):
        	print("Hello",other_cat.name, "! My name is ", self.name, "!")
