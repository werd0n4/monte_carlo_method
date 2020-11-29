from os import listdir

a = dict()
for filename in listdir():
    with open(filename) as file:
            a[filename] = [ x.strip() for x in file.readlines()]


x = ';'.join(a.keys()) + '\n'

for i in range(50):
    x += ';'.join(a[key][i] for key in a.keys()) + '\n'

with open("x.csv", 'w') as file:
    file.write(x)