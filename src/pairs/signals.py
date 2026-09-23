import numpy as np 

def generate_positions(z, entry_z=2.0, exit_z=0.5, stop_z=4.0):
    positions = np.zeros(len(z))
    current = 0
    for t in range(len(z)):
        if current == 0:
            if z[t] > entry_z:
                current = -1
            elif z[t] < -entry_z:
                current = 1
        elif current == 1 and (z[t] > -exit_z or z[t] < -stop_z):
            current = 0
        elif current == -1 and (z[t] < exit_z or z[t] > stop_z):
            current = 0
        positions[t] = current
    return positions