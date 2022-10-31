

list = [x*x for x in range(101)]

print(list)

newlist = [y for y in list if y%2 == 0]

print(newlist)
