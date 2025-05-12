import time
import numpy as np
import matplotlib.pyplot as plt


class CoreBuilder:
    @staticmethod
    def assembly_maker(assembly_ID, fuel_temp, mod_temp, bor, dep):
        """_summary_
        Used internally to create a 3D numpy array of the assembly ID and its characteristics.
        """
        assembly = np.empty([10, 10, 20], dtype=object)
        for i in range(10):
            for j in range(10):
                for k in range(20):
                    if k < 5:
                        assembly[i, j, k] = np.array(["BOTREF", fuel_temp, mod_temp, bor, 0])
                    elif k > 14:
                        assembly[i, j, k] = np.array(["TOPREF", fuel_temp, mod_temp, bor, 0])
                    else:
                        assembly[i, j, k] = np.array([assembly_ID, fuel_temp, mod_temp, bor, dep])
        print(assembly.shape)
        return assembly

    @staticmethod
    def core_maker(core_map):
        """_summary_

        Args:
            core_map: The name of the config file (without .txt)

        Returns:
            core: 3D numpy array of the characteristics needed to pull cross sections at each i, j, k location
        """
        core = np.empty([90, 90, 20], dtype=object)
        with open(f"{core_map}.txt", 'r') as map_file:
            lines = map_file.readlines()
            assembly_list = []
            fuel_temp_list = []
            mod_temp_list = []
            dep_list = []
            bor_list = []

            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    continue
                parts = line.split()
                assembly_list.append(parts[0])
                fuel_temp_list.append(parts[1])
                mod_temp_list.append(parts[2])
                bor_list.append(parts[3])
                dep_list.append(parts[4])

            n = 0
            for i in range(int(np.sqrt(len(assembly_list)))):
                for j in range(int(np.sqrt(len(assembly_list)))):
                    core[i*10:(i+1)*10, j*10:(j+1)*10, :] = CoreBuilder.assembly_maker(
                        assembly_list[n],
                        fuel_temp_list[n],
                        mod_temp_list[n],
                        bor_list[n],
                        dep_list[n]
                    )
                    n += 1
            print(core.shape)
        return core

    @staticmethod
    def config_maker(map_name, assembly_map, fuel_temp, mod_temp, boron):
        """
        Creates a configuration file for a defined core assembly map.

        Args:
            map_name (_type_): Name of the config file to be created.
            assembly_map (_type_): The core assembly map, needs assembly ID and burnup
            fuel_temp (_type_): fuel temperature in K
            mod_temp (_type_): moderator temperature in K
            boron (_type_): boron concentration in ppm
        """
        
        with open(f"{map_name}.txt", 'w') as config_file:
            n = 0
            for i in range(int(np.sqrt(len(assembly_map)))):
                for j in range(int(np.sqrt(len(assembly_map)))):
                    config_file.write(f"{assembly_map[n][0]} {fuel_temp} {mod_temp} {boron} {assembly_map[n][1]}\n")
                    n += 1
        return