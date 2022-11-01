
#limit = 3 

#def generator(limit):
     
#	list = range(limit)
#	for i in list:
#		yield i

		


#generator = generator(limit)
#for i in generator:
#	for j in generator:
#		print("Meow ")

		
		
		
import sys
def myrange(start, stop):
    #print('myrange start')
    x = start
    while x <= stop:
        #print(f'myrange yield {x}')
        
        yield x
        for i in range (x):
        	sys.stdout.write("Meow ")
        x = x + 1
    #print('myrange end')

r = myrange(1, 10)
#print(f"r = {r}")

for x in r:
    #print('Loop processing {}'.format(x))
    print(' ')

