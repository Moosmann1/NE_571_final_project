import time
import numpy as np
import matplotlib.pyplot as plt


def assembly_maker(assembly_num, dep, bor):
    assembly = np.empty([10,10,20], dtype=object)
    
    for i in range(10):
        for j in range(10):
            for k in range(20):
                if k < 5:
                    assembly[i,j,k] = np.array([1, 0, bor])
                elif k > 15:
                    assembly[i,j,k] = np.array([2, 0, bor])
                else:
                    assembly[i,j,k] = np.array([assembly_num, dep, bor])

    return assembly


def core_maker(core_map):
    core = np.empty([90,90,20], dtype=object)
    
    with open(f"{core_map}.txt", 'r') as map_file:
        lines = map_file.readlines()
        assembly_list = []
        dep_list = []
        bor_list = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split()
            assembly_list.append(int(parts[0]))
            dep_list.append(int(parts[1]))
            bor_list.append(int(parts[2]))
            
        n = 0
        for i in range(9):
            for j in range(9):
                core[i*10:(i+1)*10, j*10:(j+1)*10, :] = assembly_maker(assembly_list[n], dep_list[n], bor_list[n])
                n += 1
    return core

core_test = core_maker("core_test")
    
