from cat import Cat


if __name__ == "__main__":
	cat1 = Cat("name1")  
	cat2 = Cat("name2")
	cat1.greet(cat2)
	cat2.greet(cat1)
