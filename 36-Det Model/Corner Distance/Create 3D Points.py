from math import cos, sin, pi, sqrt

r = 7.1
o = .7688446995
list1 = []
for i in range(0, 360, 10):
    i = i * 2 * pi / 360
    y = 6 * sin(i)
    z = r * cos(i) * sqrt(2) / 2 + o
    x = r * cos(i) * sqrt(2) / 2 + o

    # x + 13.35, y, z - 1.65 for corner2
    # x + 0.05, y, z + 0.05 for corner1
    list1.append([x + 13.45, y, z - 1.55])
    #print('{:0.2f} {:1.2f} {:2.2f}'.format(x+0.05,y,z+0.05))


list2 = []
with open("DetOrder", "r") as data:
    for line in data:
        list2.append(int(line))

zipped_lists = zip(list2, list1)
sorted_zipped_lists = sorted(zipped_lists)
sorted_list1 = [element for _, element in sorted_zipped_lists]

InputDetOrder = []
with open("Order", "r") as data:
    for line in data:
        InputDetOrder.append(sorted_list1[int(line)-1])

for i in InputDetOrder:
    print('{:0.2f} {:1.2f} {:2.2f}'.format(i[0], i[1], i[2]))
