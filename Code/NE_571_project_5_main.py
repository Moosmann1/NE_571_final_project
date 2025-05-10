import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu
from core_builder import CoreBuilder
from extract import CrossSectionVocabulary

# Update grand_xs_library to include all cross section dataframes when they exist
# Note the placeholder .csv used. So reflectors are not included yet.
grand_xs_library = {
    "NUu250c00": CrossSectionVocabulary("XS.csv"),
    "rad":       CrossSectionVocabulary("XS.csv"),
    "BOTREF":    CrossSectionVocabulary("XS.csv"),
    "TOPREF":    CrossSectionVocabulary("XS.csv"),
}


# Generates the matrixes for the flux search. 
def matrix(core_map, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim):
    """
    Generates the matrices for the flux search in a nuclear reactor core.
        
    Args:
        core_map (numpy.ndarray): A 3D array representing the reactor core map. 
            Each element is a tuple where the first value indicates the region type 
            (e.g., "BOTREF", "TOPREF", or fuel region), and the remaining values 
            provide cross-section parameters.
        assembly_ij_dim (float): The physical dimension of the assembly in the i and j directions (in cm).
        fuel_k_dim (float): The physical height of the fuel region in the k direction (in cm).
        bottom_ref_k_dim (float): The physical height of the bottom reflector region in the k direction (in cm).
        top_ref_k_dim (float): The physical height of the top reflector region in the k direction (in cm).

    Notes:
        - The function calculates the step sizes in the i, j, and k directions based on the core map dimensions 
          and the physical dimensions of the regions.
        - The matrices are constructed using the cross-section data from `grand_xs_library`
          and are formatted for sparce LU decomposition.
    """
    #track time to make the matrix
    start_time = time.time()
    
    # Get the total number of nodes in each direction
    i_node_total = core_map.shape[0]
    j_node_total = core_map.shape[1]
    k_node_total = core_map.shape[2]

    # Calculate step sizes in the i and j directions (uniform for all regions)
    di_core = (9*assembly_ij_dim) / i_node_total
    dj_core = (9*assembly_ij_dim) / j_node_total
    print(di_core)
    print(dj_core)
    # Count nodes in the k direction for each region
    k_nodes_bottom_ref = 0
    k_nodes_fuel = 0
    k_nodes_top_ref = 0

    for k in range(k_node_total):
        if core_map[0, 0, k][0] == "BOTREF":
            k_nodes_bottom_ref += 1
        elif core_map[0, 0, k][0] == "TOPREF":
            k_nodes_top_ref += 1
        else:
            k_nodes_fuel += 1
    
    # Calculate step sizes in the k direction for each region
    dk_bottom_ref = bottom_ref_k_dim / k_nodes_bottom_ref if k_nodes_bottom_ref > 0 else 0
    dk_fuel = fuel_k_dim / k_nodes_fuel if k_nodes_fuel > 0 else 0
    dk_top_ref = top_ref_k_dim / k_nodes_top_ref if k_nodes_top_ref > 0 else 0

    print(dk_bottom_ref)
    print(dk_fuel)
    print(dk_top_ref)
    
    # A matrixes: the 1-D arrays that will get diagonalized
    central_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    east_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    west_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    north_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    south_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    up_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    down_fast = np.zeros(i_node_total*j_node_total*k_node_total)
    
    central_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    east_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    west_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    north_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    south_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    up_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    down_thermal = np.zeros(i_node_total*j_node_total*k_node_total)
    
    #B matrixes: the 1-D arrays that will get diagonalized
    central_B1_fast_fission = np.zeros(i_node_total*j_node_total*k_node_total)
    central_B1_thermal_fission = np.zeros(i_node_total*j_node_total*k_node_total)
    central_B2 = np.zeros(i_node_total*j_node_total*k_node_total)
    
    it = 0
    for k in range(core_map.shape[2]):
        for j in range(core_map.shape[1]):
            for i in range(core_map.shape[0]):
                 # Determine the correct dk value based on the region
                if core_map[i, j, k][0] == "BOTREF":
                    dk = dk_bottom_ref
                elif core_map[i, j, k][0] == "TOPREF":
                    dk = dk_top_ref
                else:
                    dk = dk_fuel

                # Get the cross-section data for the central node
                fast_xs_central = grand_xs_library[core_map[i, j, k][0]].get(
                    float(core_map[i, j, k][1]),
                    float(core_map[i, j, k][2]),
                    float(core_map[i, j, k][3]),
                    float(core_map[i, j, k][4]),
                    1.0
                )
                thermal_xs_central = grand_xs_library[core_map[i, j, k][0]].get(
                    float(core_map[i, j, k][1]),
                    float(core_map[i, j, k][2]),
                    float(core_map[i, j, k][3]),
                    float(core_map[i, j, k][4]),
                    2.0
                )

                # Populate central diagonal
                central_fast[it] = (
                    (2 * fast_xs_central["DIFF(1/3TR)"] / di_core**2) +
                    (2 * fast_xs_central["DIFF(1/3TR)"] / dj_core**2) +
                    (2 * fast_xs_central["DIFF(1/3TR)"] / dk**2) +
                    (fast_xs_central["ABSORPTION"] + fast_xs_central["OUT-SCATTER"])
                )
                central_thermal[it] = (
                    (2 * thermal_xs_central["DIFF(1/3TR)"] / di_core**2) +
                    (2 * thermal_xs_central["DIFF(1/3TR)"] / dj_core**2) +
                    (2 * thermal_xs_central["DIFF(1/3TR)"] / dk**2) +
                    thermal_xs_central["ABSORPTION"]
                )
                
                # Populate B matrixes
                central_B1_fast_fission[it] = fast_xs_central["NU-FISSION"]
                central_B1_thermal_fission[it] = thermal_xs_central["NU-FISSION"]
                central_B2[it] = fast_xs_central["OUT-SCATTER"]

                # Handle off-diagonals (e.g., west, east, etc.)
                if i > 0:  # West
                    # cross sections
                    fast_xs_west = grand_xs_library[core_map[i-1, j, k][0]].get(
                        float(core_map[i-1, j, k][1]), 
                        float(core_map[i-1, j, k][2]), 
                        float(core_map[i-1, j, k][3]), 
                        float(core_map[i-1, j, k][4]), 
                        1.0)
                    thermal_xs_west = grand_xs_library[core_map[i-1, j, k][0]].get(
                        float(core_map[i-1, j, k][1]), 
                        float(core_map[i-1, j, k][2]), 
                        float(core_map[i-1, j, k][3]), 
                        float(core_map[i-1, j, k][4]), 
                        2.0)
                    
                    # populatre the matrix diagonals
                    west_fast[it] = -1 * fast_xs_west["DIFF(1/3TR)"] / di_core**2
                    west_thermal[it] = -1 * thermal_xs_west["DIFF(1/3TR)"] / di_core**2

                if i < core_map.shape[0] - 1:  # East
                    # cross sections
                    fast_xs_east = grand_xs_library[core_map[i+1, j, k][0]].get(
                        float(core_map[i+1, j, k][1]), 
                        float(core_map[i+1, j, k][2]), 
                        float(core_map[i+1, j, k][3]), 
                        float(core_map[i+1, j, k][4]), 
                        1.0)
                    thermal_xs_east = grand_xs_library[core_map[i+1, j, k][0]].get(
                        float(core_map[i+1, j, k][1]), 
                        float(core_map[i+1, j, k][2]), 
                        float(core_map[i+1, j, k][3]), 
                        float(core_map[i+1, j, k][4]), 
                        2.0)
                    
                    # populatre the matrix diagonals
                    east_fast[it] = -1 * fast_xs_east["DIFF(1/3TR)"] / di_core**2
                    east_thermal[it] = -1 * thermal_xs_east["DIFF(1/3TR)"] / di_core**2
                
                if j > 0:  # South
                    # cross sections
                    fast_xs_south = grand_xs_library[core_map[i, j - 1, k][0]].get(
                        float(core_map[i, j - 1, k][1]),
                        float(core_map[i, j - 1, k][2]),
                        float(core_map[i, j - 1, k][3]),
                        float(core_map[i, j - 1, k][4]),
                        1.0
                    )
                    thermal_xs_south = grand_xs_library[core_map[i, j - 1, k][0]].get(
                        float(core_map[i, j - 1, k][1]),
                        float(core_map[i, j - 1, k][2]),
                        float(core_map[i, j - 1, k][3]),
                        float(core_map[i, j - 1, k][4]),
                        2.0
                    )
                    
                    # populatre the matrix diagonals
                    south_fast[it] = -1 * fast_xs_south["DIFF(1/3TR)"] / dj_core**2
                    south_thermal[it] = -1 * thermal_xs_south["DIFF(1/3TR)"] / dj_core**2

                if j < core_map.shape[1] - 1:  # North
                    # cross sections
                    fast_xs_north = grand_xs_library[core_map[i, j+1, k][0]].get(
                        float(core_map[i, j+1, k][1]), 
                        float(core_map[i, j+1, k][2]), 
                        float(core_map[i, j+1, k][3]), 
                        float(core_map[i, j+1, k][4]), 
                        1.0)
                    thermal_xs_north = grand_xs_library[core_map[i, j+1, k][0]].get(
                        float(core_map[i, j+1, k][1]), 
                        float(core_map[i, j+1, k][2]), 
                        float(core_map[i, j+1, k][3]), 
                        float(core_map[i, j+1, k][4]), 
                        2.0)
                    
                    # populatre the matrix diagonals
                    north_fast[it] = -1 * fast_xs_north["DIFF(1/3TR)"] / dj_core**2
                    north_thermal[it] = -1 * thermal_xs_north["DIFF(1/3TR)"] / dj_core**2
                
                if k > 0:  # Down
                    # cross sections
                    fast_xs_down = grand_xs_library[core_map[i, j, k-1][0]].get(
                        float(core_map[i, j, k-1][1]), 
                        float(core_map[i, j, k-1][2]), 
                        float(core_map[i, j, k-1][3]), 
                        float(core_map[i, j, k-1][4]), 
                        1.0)
                    thermal_xs_down = grand_xs_library[core_map[i, j, k-1][0]].get(
                        float(core_map[i, j, k-1][1]), 
                        float(core_map[i, j, k-1][2]), 
                        float(core_map[i, j, k-1][3]), 
                        float(core_map[i, j, k-1][4]), 
                        2.0)
                    
                    # populatre the matrix diagonals
                    down_fast[it] = -1 * fast_xs_down["DIFF(1/3TR)"] / dk**2
                    down_thermal[it] = -1 * thermal_xs_down["DIFF(1/3TR)"] / dk**2
                
                if k < core_map.shape[2] - 1:  # Up
                    # cross sections
                    fast_xs_up = grand_xs_library[core_map[i, j, k+1][0]].get(
                        float(core_map[i, j, k+1][1]), 
                        float(core_map[i, j, k+1][2]), 
                        float(core_map[i, j, k+1][3]), 
                        float(core_map[i, j, k+1][4]), 
                        1.0)
                    thermal_xs_up = grand_xs_library[core_map[i, j, k+1][0]].get(
                        float(core_map[i, j, k+1][1]), 
                        float(core_map[i, j, k+1][2]), 
                        float(core_map[i, j, k+1][3]), 
                        float(core_map[i, j, k+1][4]), 
                        2.0)
                    
                    # populatre the matrix diagonals
                    up_fast[it] = -1 * fast_xs_up["DIFF(1/3TR)"] / dk**2
                    up_thermal[it] = -1 * thermal_xs_up["DIFF(1/3TR)"] / dk**2

                print(it)
                it += 1
    
    # CSC formating for LU decomposition (sprase matrix)
    # Fast A matrix
    A1 = diags(
        [
            central_fast,
            east_fast[:-1],
            west_fast[1:],
            north_fast[:-j_node_total],
            south_fast[j_node_total:],
            up_fast[:-(j_node_total * i_node_total)],
            down_fast[(j_node_total * i_node_total):],
        ],
        [0, 1, -1, j_node_total, -j_node_total, j_node_total * i_node_total, -(j_node_total * i_node_total)],
        format="csc",
    )

    # Thermal A matrix
    A2 = diags(
        [
            central_thermal,
            east_thermal[:-1],
            west_thermal[1:],
            north_thermal[:-j_node_total],
            south_thermal[j_node_total:],
            up_thermal[:-(j_node_total * i_node_total)],
            down_thermal[(j_node_total * i_node_total):],
        ],
        [0, 1, -1, j_node_total, -j_node_total, j_node_total * i_node_total, -(j_node_total * i_node_total)],
        format="csc",
    )

    # Fast B matrices for fast and thermal fission
    B1_fast_fission = diags([central_B1_fast_fission], [0], format="csc")
    B1_thermal_fission = diags([central_B1_thermal_fission], [0], format="csc")

    # Thermal B matrix
    B2 = diags([central_B2], [0], format="csc")
    
    # time to make the matrix
    end_time = time.time()
    print(f"Time taken to create matrix: {end_time - start_time:.2f} seconds")
    return A1, A2, B1_fast_fission, B1_thermal_fission, B2

# The fluxsearch function solves the flux equations using the LU decomposition method.
def fluxsearch(A1, A2, B1f, B1t, B2): 
    """
    Perform the flux search using LU decomposition on sparse matrices.
    
    Args:
        A1 (scipy.sparse.csc_matrix): Fast A matrix (sparse matrix for fast flux equations).
        A2 (scipy.sparse.csc_matrix): Thermal A matrix (sparse matrix for thermal flux equations).
        B1f (scipy.sparse.csc_matrix): Fast fission B matrix (source term for fast flux).
        B1t (scipy.sparse.csc_matrix): Thermal fission B matrix (source term for thermal flux).
        B2 (scipy.sparse.csc_matrix): Thermal B matrix (scattering source term from fast to thermal flux).

    Returns:
       
    """
    
    start_time = time.time()
    flux1 = np.ones(A1.shape[0])
    flux2 = np.ones(A2.shape[0])
    k = 1
    diff = 1
    count = 0
    S1 = B1f.dot(flux1)
    S2 = B1t.dot(flux2)
    S = S1 + S2
    Sig_1_2 = B2.dot(flux1)
    
    # Perform LU decomposition on A1 and A2
    lu_A1 = splu(A1)
    lu_A2 = splu(A2)
    
    while diff > 0.000001:
        count += 1
        #print(count)

        oldk = k
        oldS = S
        oldSig_1_2 = Sig_1_2
        flux1 = (1 / oldk) * lu_A1.solve(oldS)
        flux2 = lu_A2.solve(oldSig_1_2)
        S1 = B1f.dot(flux1)
        S2 = B1t.dot(flux2)
        S = S1 + S2
        Sig_1_2 = B2.dot(flux1)
        k = np.sum(S) / (np.sum(oldS) * (1 / oldk))
        diff = np.sqrt((k - oldk)**2)

    
    end_time = time.time()
    print(f"Time taken to find k: {end_time - start_time:.2f} seconds")
    return k, flux1, flux2


# old norm function from my second project. Maybe you can rework it for this project. or just develop a new one.
def norm(flux1, flux2, power, i_core_dim, j_core_dim, k_core_dim, i_core_nodes, j_core_nodes, k_core_nodes, ref_nodes):
    """
    Calculate a normalized flux to some power level.

    Parameters
    ----------
    flux : np.ndarray
        Unnormalized flux.
    power : float
        Power level to normalize the flux to (MW).
    i_dim : float
        Length of the domain in the x direction.
    j_dim : float
        Length of the domain in the y direction.
    k_dim : float
        Length of the domain in the z direction.

    Returns
    -------
    float
        Norm of the flux.
    """
    energy_per_fission = 3.2E-11  # Joule/fission
    
    i_nodes = (i_core_nodes + 2*ref_nodes)
    j_nodes = (j_core_nodes + 2*ref_nodes)
    k_nodes = (k_core_nodes + 2*ref_nodes)
    
    intermediate_flux1 = np.zeros((i_nodes * j_nodes * k_nodes))
    intermediate_flux2 = np.zeros((i_nodes * j_nodes * k_nodes))
    
    it2 = 0
    for k in range(k_nodes):
        for j in range(j_nodes):
            for i in range(i_nodes):
                intermediate_flux1[it2] = g1_neusigf[material_matrix_local[i, j, k]] * flux1[it2]
                intermediate_flux2[it2] = g2_neusigf[material_matrix_local[i, j, k]] * flux2[it2]
                it2 += 1
    norm_const = (power*1E6) / (energy_per_fission * (np.sum(intermediate_flux1) + np.sum(intermediate_flux2)) * (i_core_dim * j_core_dim * k_core_dim))
    
    # Reshape and pad flux1
    
    n = round(i_nodes)
    flux1_reshape = flux1.reshape((n, n, n))  # only works for cubes
    flux1_reshape = np.pad(flux1_reshape, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    flux1 = flux1_reshape.flatten()
    
    # Reshape and pad flux2
    n = round(i_nodes)
    flux2_reshape = flux2.reshape((n, n, n))  # only works for cubes
    flux2_reshape = np.pad(flux2_reshape, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    flux2 = flux2_reshape.flatten()
    
    return norm_const * flux1, norm_const * flux2

# testing. No reflector cross sections yet. 

# load the core map from the config file
core_test = CoreBuilder.core_maker("core_map2")

# placeholder dimensions. Will need to find the dimensions from NuScale documentation
#should be good enough to start thinking of part 5 and 6
assembly_ij_dim = 10  # assembly dimensions cm (sysmentrical)
fuel_k_dim = 200  # fuel region height cm
bottom_ref_k_dim = 10 # bottom reflector height cm (may be hard to find...)
top_ref_k_dim = 10 # top reflector height cm (may be hard to find...)

A1, A2, B1_fast_fission, B1_thermal_fission, B2 = matrix(core_test, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim)

# Call the fluxsearch function
k_1, flux1_1, flux2_1 = fluxsearch(A1, A2, B1_fast_fission, B1_thermal_fission, B2)
print(k_1)
