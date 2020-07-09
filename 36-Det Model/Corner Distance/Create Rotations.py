from math import cos, sin, pi, sqrt
import numpy as np

for i in range(0, 360, 10):
    i = i * 2 * pi / 360

    z_rot = np.array([[cos(i), -sin(i), 0],
                      [sin(i), cos(i), 0],
                      [0, 0, 1]])

    y_rot = np.array([[cos(pi/4), 0, sin(pi/4)],
                      [0, 1, 0],
                      [-sin(pi/4), 0, cos(pi/4)]])

    data = np.around(np.matmul(z_rot, y_rot), decimals=3).flatten().tolist()
    print(*data, sep=' ')

