import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu
from core_builder import CoreBuilder
from extract import CrossSectionVocabulary

def get(self, tf, tm, bor, burnup, group):
    key = (burnup, tf, tm, bor, group)
    if key in self.vocab:
        return self.vocab[key]
    else:
        raise KeyError(f"No cross section data found for: BURNUP={burnup}, TF={tf}, TM={tm}, BOR={bor}, GROUP={group}")
 

# Generates the matrixes for the flux search. 
def matrix(grand_library, core_map, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim):
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
                fast_xs_central = get(grand_library[core_map[i, j, k][0]],
                    float(core_map[i, j, k][1]),
                    float(core_map[i, j, k][2]),
                    float(core_map[i, j, k][3]),
                    float(core_map[i, j, k][4]),
                    1.0
                )
                thermal_xs_central = get(grand_library[core_map[i, j, k][0]],
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
                    fast_xs_west = get(grand_library[core_map[i-1, j, k][0]],
                        float(core_map[i-1, j, k][1]), 
                        float(core_map[i-1, j, k][2]), 
                        float(core_map[i-1, j, k][3]), 
                        float(core_map[i-1, j, k][4]), 
                        1.0)
                    thermal_xs_west = get(grand_library[core_map[i-1, j, k][0]],
                        float(core_map[i-1, j, k][1]), 
                        float(core_map[i-1, j, k][2]), 
                        float(core_map[i-1, j, k][3]), 
                        float(core_map[i-1, j, k][4]), 
                        2.0)
                    
                    # populate the matrix diagonals
                    west_fast[it] = -1 * fast_xs_west["DIFF(1/3TR)"] / di_core**2
                    west_thermal[it] = -1 * thermal_xs_west["DIFF(1/3TR)"] / di_core**2

                if i < core_map.shape[0] - 1:  # East
                    # cross sections
                    fast_xs_east = get(grand_library[core_map[i+1, j, k][0]],
                        float(core_map[i+1, j, k][1]), 
                        float(core_map[i+1, j, k][2]), 
                        float(core_map[i+1, j, k][3]), 
                        float(core_map[i+1, j, k][4]), 
                        1.0)
                    thermal_xs_east = get(grand_library[core_map[i+1, j, k][0]],
                        float(core_map[i+1, j, k][1]), 
                        float(core_map[i+1, j, k][2]), 
                        float(core_map[i+1, j, k][3]), 
                        float(core_map[i+1, j, k][4]), 
                        2.0)
                    
                    # populate the matrix diagonals
                    east_fast[it] = -1 * fast_xs_east["DIFF(1/3TR)"] / di_core**2
                    east_thermal[it] = -1 * thermal_xs_east["DIFF(1/3TR)"] / di_core**2
                
                if j > 0:  # South
                    # cross sections
                    fast_xs_south = get(grand_library[core_map[i, j - 1, k][0]],
                        float(core_map[i, j - 1, k][1]),
                        float(core_map[i, j - 1, k][2]),
                        float(core_map[i, j - 1, k][3]),
                        float(core_map[i, j - 1, k][4]),
                        1.0
                    )
                    thermal_xs_south = get(grand_library[core_map[i, j - 1, k][0]],
                        float(core_map[i, j - 1, k][1]),
                        float(core_map[i, j - 1, k][2]),
                        float(core_map[i, j - 1, k][3]),
                        float(core_map[i, j - 1, k][4]),
                        2.0
                    )
                    
                    # populate the matrix diagonals
                    south_fast[it] = -1 * fast_xs_south["DIFF(1/3TR)"] / dj_core**2
                    south_thermal[it] = -1 * thermal_xs_south["DIFF(1/3TR)"] / dj_core**2

                if j < core_map.shape[1] - 1:  # North
                    # cross sections
                    fast_xs_north = get(grand_library[core_map[i, j+1, k][0]],
                        float(core_map[i, j+1, k][1]), 
                        float(core_map[i, j+1, k][2]), 
                        float(core_map[i, j+1, k][3]), 
                        float(core_map[i, j+1, k][4]), 
                        1.0)
                    thermal_xs_north = get(grand_library[core_map[i, j+1, k][0]],
                        float(core_map[i, j+1, k][1]), 
                        float(core_map[i, j+1, k][2]), 
                        float(core_map[i, j+1, k][3]), 
                        float(core_map[i, j+1, k][4]), 
                        2.0)
                    
                    # populate the matrix diagonals
                    north_fast[it] = -1 * fast_xs_north["DIFF(1/3TR)"] / dj_core**2
                    north_thermal[it] = -1 * thermal_xs_north["DIFF(1/3TR)"] / dj_core**2
                
                if k > 0:  # Down
                    # cross sections
                    fast_xs_down = get(grand_library[core_map[i, j, k-1][0]],
                        float(core_map[i, j, k-1][1]), 
                        float(core_map[i, j, k-1][2]), 
                        float(core_map[i, j, k-1][3]), 
                        float(core_map[i, j, k-1][4]), 
                        1.0)
                    thermal_xs_down = get(grand_library[core_map[i, j, k-1][0]],
                        float(core_map[i, j, k-1][1]), 
                        float(core_map[i, j, k-1][2]), 
                        float(core_map[i, j, k-1][3]), 
                        float(core_map[i, j, k-1][4]), 
                        2.0)
                    
                    # populate the matrix diagonals
                    down_fast[it] = -1 * fast_xs_down["DIFF(1/3TR)"] / dk**2
                    down_thermal[it] = -1 * thermal_xs_down["DIFF(1/3TR)"] / dk**2
                
                if k < core_map.shape[2] - 1:  # Up
                    # cross sections
                    fast_xs_up = get(grand_library[core_map[i, j, k+1][0]],
                        float(core_map[i, j, k+1][1]), 
                        float(core_map[i, j, k+1][2]), 
                        float(core_map[i, j, k+1][3]), 
                        float(core_map[i, j, k+1][4]), 
                        1.0)
                    thermal_xs_up = get(grand_library[core_map[i, j, k+1][0]],
                        float(core_map[i, j, k+1][1]), 
                        float(core_map[i, j, k+1][2]), 
                        float(core_map[i, j, k+1][3]), 
                        float(core_map[i, j, k+1][4]), 
                        2.0)
                    
                    # populate the matrix diagonals
                    up_fast[it] = -1 * fast_xs_up["DIFF(1/3TR)"] / dk**2
                    up_thermal[it] = -1 * thermal_xs_up["DIFF(1/3TR)"] / dk**2

                # print(it)
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

# === Normalize flux and plot power/flux maps ===
def normalize_and_plot(flux1, flux2, core_map, power_MW, assembly_dim_cm, fuel_height_cm):
    import matplotlib.pyplot as plt
    import numpy as np

    energy_per_fission = 3.2e-11  # J/fission
    node_vol_cm3 = assembly_dim_cm**2 * (fuel_height_cm / 10)  # volume per node in fuel region (cm^3)

    fission_source = []
    power_map = []
    flux_map = []

    for ai in range(9):  # 9x9 assemblies
        row_flux = []
        row_power = []
        for aj in range(9):
            local_flux1 = 0
            local_flux2 = 0
            local_power = 0

            for i in range(ai*10, (ai+1)*10):
                for j in range(aj*10, (aj+1)*10):
                    for k in range(5, 15):  # fuel nodes only
                        flat_idx = i + 90*j + 90*90*k
                        entry = core_map[i, j, k]
                        mat = entry[0]
                        tf = float(entry[1])
                        tm = float(entry[2])
                        bor = float(entry[3])
                        dep = float(entry[4])

                        xs1 = grand_xs_library[mat].get(tf, tm, bor, dep, 1.0)
                        xs2 = grand_xs_library[mat].get(tf, tm, bor, dep, 2.0)

                        sigf1 = xs1["FISSION"]
                        sigf2 = xs2["FISSION"]

                        local_flux1 += flux1[flat_idx]
                        local_flux2 += flux2[flat_idx]
                        local_power += (
                            flux1[flat_idx] * sigf1 + flux2[flat_idx] * sigf2
                        ) * node_vol_cm3 * energy_per_fission

                        fission_source.append((flux1[flat_idx] * sigf1 + flux2[flat_idx] * sigf2) * node_vol_cm3)

            avg_flux = (local_flux1 + local_flux2) / (10 * 10 * 10) #possibly change to 10 * 10 so as to get the ij average over sum of k fluxes
            row_flux.append(avg_flux)
            row_power.append(local_power)
        flux_map.append(row_flux)
        power_map.append(row_power)

    # Normalize power map to match target power
    total_power_W = power_MW * 1e6
    raw_total_power = np.sum(power_map)
    norm_factor = total_power_W / raw_total_power
    flux_map = (np.array(flux_map) * norm_factor).tolist()
    power_map = (np.array(power_map) * norm_factor).tolist()

    # === Plot Flux Map ===
    plt.figure(figsize=(7, 6))
    plt.imshow(flux_map, cmap='viridis')
    plt.title("Average Assembly Flux (normalized units)")
    plt.xlabel("X Assembly Index")
    plt.ylabel("Y Assembly Index")
    plt.colorbar(label="Flux")
    plt.tight_layout()
    plt.show()

    # === Plot Power Map ===
    plt.figure(figsize=(7, 6))
    plt.imshow(power_map, cmap='inferno')
    plt.title("Average Assembly Power (W)")
    plt.xlabel("X Assembly Index")
    plt.ylabel("Y Assembly Index")
    plt.colorbar(label="Power (W)")
    plt.tight_layout()
    plt.show()

    # === 1D Line Plots Along Middle Row ===
    middle_row_flux = flux_map[4]  # middle row (Y = 4)
    middle_row_power = power_map[4]
    x = list(range(len(middle_row_flux)))  # X assembly indices

    # Line plot for flux
    plt.figure()
    plt.plot(x, middle_row_flux, marker='o')
    plt.title("Flux Along Middle Row (Y = 4)")
    plt.xlabel("X Assembly Index")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Line plot for power
    plt.figure()
    plt.plot(x, middle_row_power, marker='o', color='r')
    plt.title("Power Along Middle Row (Y = 4)")
    plt.xlabel("X Assembly Index")
    plt.ylabel("Power (W)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Axial Flux Plot Along Middle Column (Z Direction) ===
    # === Axial Flux Plot Along Middle Column ===
    mid_i = 45
    mid_j = 45
    axial_flux = []
    axial_power = []

    for k in range(20):  # loop over all axial layers
        flat_idx = mid_i + 90 * mid_j + 90 * 90 * k
        entry = core_map[mid_i, mid_j, k]
        mat = entry[0]
        tf = float(entry[1])
        tm = float(entry[2])
        bor = float(entry[3])
        dep = float(entry[4])

        xs1 = grand_xs_library[mat].get(tf, tm, bor, dep, 1.0)
        xs2 = grand_xs_library[mat].get(tf, tm, bor, dep, 2.0)

        sigf1 = xs1["FISSION"]
        sigf2 = xs2["FISSION"]

        phi1 = flux1[flat_idx]
        phi2 = flux2[flat_idx]
        total_flux = phi1 + phi2
        total_power = (phi1 * sigf1 + phi2 * sigf2) * node_vol_cm3 * energy_per_fission

        axial_flux.append(total_flux)
        axial_power.append(total_power)

    # Plot axial flux
    dr_bottom_ref = bottom_ref_k_dim / 5
    dr_fuel = fuel_k_dim / 10
    dr_top_ref = top_ref_k_dim / 5
    physical_steps = [0, 1*dr_bottom_ref, 2*dr_bottom_ref, 3*dr_bottom_ref, 4*dr_bottom_ref,5*dr_fuel, 6*dr_fuel, 7*dr_fuel, 8*dr_fuel, 9*dr_fuel, 10*dr_fuel, 11*dr_fuel, 12*dr_fuel, 13*dr_fuel, 14*dr_fuel, 15*dr_top_ref, 16*dr_top_ref, 17*dr_top_ref, 18*dr_top_ref, 19*dr_top_ref]
    
    plt.figure()
    plt.plot(physical_steps, axial_flux, marker='s')
    plt.title("Axial Flux Along Central Column (i=45, j=45)")
    plt.xlabel("Z Node Index")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot axial power
    # Plot axial power
    plt.figure()
    plt.plot(range(0, 20), axial_power, marker='s', color='orange')
    plt.title("Axial Power Along Central Column (i=45, j=45)")
    plt.xlabel("Z Node Index")
    plt.ylabel("Power (W)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def bisection_boron(core_name, assembly_map, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim, boron_low=600, boron_high=1800, tol=1e-5, max_iter=100):
    """
    Perform bisection to adjust boron concentration to achieve critical k_eff (k_eff ≈ 1.0).

    Args:
        core_map (str): Name of the core map file.
        assembly_map (list): Assembly map configuration.
        assembly_ij_dim (float): XY dimension per assembly (cm).
        fuel_k_dim (float): Axial height of fuel (cm).
        bottom_ref_k_dim (float): Height of bottom reflector (cm).
        top_ref_k_dim (float): Height of top reflector (cm).
        thermal_power (float): Reactor thermal power (MW).
        tol (float): Tolerance for k_eff to be considered critical (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: Critical boron concentration.
        float: Final k_eff value.
    """

    boron_critical = None

    # Initialize CoreBuilder
    builder = CoreBuilder()

    for iteration in range(max_iter):
        # Calculate midpoint boron concentration
        boron_mid = (boron_low + boron_high) / 2

        # Write a new configuration file with the updated boron concentration
        builder.config_maker(core_name, assembly_map, fuel_temp=1019.3, mod_temp=557, boron=boron_mid)

        # Generate the core and matrices
        interpolated_xs_library = interpolate_xs(grand_xs_library, boron_mid)

        core = CoreBuilder.core_maker(core_name)
        A1, A2, B1_fast_fission, B1_thermal_fission, B2 = matrix(interpolated_xs_library, core, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim)

        # Perform flux search to calculate k_eff
        k_eff, flux1, flux2 = fluxsearch(A1, A2, B1_fast_fission, B1_thermal_fission, B2)

        print(f"Iteration {iteration + 1}: Boron = {boron_mid:.2f} ppm, k_eff = {k_eff:.6f}")

        # Check if k_eff is critical
        if abs(k_eff - 1.0) < tol:
            boron_critical = boron_mid
            print(f"Critical boron concentration found: {boron_critical:.2f} ppm, k_eff = {k_eff:.6f}")
            break

        # Update boron bounds based on k_eff
        if k_eff > 1.0:
            boron_low = boron_mid  # Increase boron
        else:
            boron_high = boron_mid  # Decrease boron

    # If maximum iterations reached without convergence
    if boron_critical is None:
        print("Bisection method did not converge to a critical boron concentration.")
        boron_critical = boron_mid

    return boron_critical, k_eff, flux1, flux2

# # === Import necessary libraries ===
# Update grand_xs_library to include all cross section dataframes when they exist
# Note the placeholder .csv used. So reflectors are not included yet.
grand_xs_library = {
    "NUu260c00": CrossSectionVocabulary("../Cross Section Data/XS_excel/XS260.csv"),
    "NUu405c00": CrossSectionVocabulary("../Cross Section Data/XS_excel/XS405.csv"),
    "NUu455c50": CrossSectionVocabulary("../Cross Section Data/XS_excel/XS455.csv"),
    "rad":       CrossSectionVocabulary("../Cross Section Data/XS_excel/radref.csv"),
    "BOTREF":    CrossSectionVocabulary("../Cross Section Data/XS_excel/botref.csv"),
    "TOPREF":    CrossSectionVocabulary("../Cross Section Data/XS_excel/topref.csv"),
}

import pandas as pd
def interpolate_xs(grand_xs_library_loc, boron):
    for material, vocab in grand_xs_library_loc.items():
        # Get the DataFrame from the CrossSectionVocabulary object
        df = vocab.df

        # Ensure BOR is a float for consistent comparisons
        df['BOR'] = df['BOR'].astype(float)

        # Check if the boron value already exists
        if boron in df['BOR'].values:
            print(f"Boron value {boron} ppm already exists for material {material}. Skipping interpolation.")
            continue

        # Perform interpolation for each unique combination of BURNUP, TF, TM, and GROUP
        interpolated_rows = []
        for (burnup, tf, tm, group), group_df in df.groupby(['BURNUP', 'TF', 'TM', 'GROUP']):
            # Sort by BOR for interpolation
            group_df = group_df.sort_values(by='BOR')

            # Find the closest BOR values for interpolation
            lower_row = group_df[group_df['BOR'] <= boron].iloc[-1] if not group_df[group_df['BOR'] <= boron].empty else None
            upper_row = group_df[group_df['BOR'] >= boron].iloc[0] if not group_df[group_df['BOR'] >= boron].empty else None

            if lower_row is None or upper_row is None:
                print(f"Cannot interpolate for BOR={boron} ppm in material {material} for BURNUP={burnup}, TF={tf}, TM={tm}, GROUP={group}.")
                continue

            # Perform linear interpolation for all cross-section columns
            interpolated_data = {'BOR': boron, 'BURNUP': burnup, 'TF': tf, 'TM': tm, 'GROUP': group}
            for column in ['ABSORPTION', 'CAPTURE', 'FISSION', 'NU-FISSION', 'TRANSPORT', 'OUT-SCATTER', 'DIFF(1/3TR)']:
                interpolated_data[column] = lower_row[column] + (
                    (upper_row[column] - lower_row[column]) * (boron - lower_row['BOR']) / (upper_row['BOR'] - lower_row['BOR'])
                )

            interpolated_rows.append(interpolated_data)

        # Add the interpolated rows to the DataFrame
        if interpolated_rows:
            interpolated_df = pd.DataFrame(interpolated_rows)
            vocab.df = pd.concat([df, interpolated_df], ignore_index=True).sort_values(by=['BURNUP', 'TF', 'TM', 'GROUP', 'BOR'])

            # Update the vocab dictionary
            for _, row in interpolated_df.iterrows():
                key = (row["BURNUP"], row["TF"], row["TM"], row["BOR"], row["GROUP"])
                vocab.vocab[key] = {
                    "ABSORPTION": row["ABSORPTION"],
                    "CAPTURE": row["CAPTURE"],
                    "FISSION": row["FISSION"],
                    "NU-FISSION": row["NU-FISSION"],
                    "TRANSPORT": row["TRANSPORT"],
                    "OUT-SCATTER": row["OUT-SCATTER"],
                    "DIFF(1/3TR)": row["DIFF(1/3TR)"],
                }

    return grand_xs_library_loc


# # === Load the core map from config file ===
core_2 = CoreBuilder.core_maker("core_map2") # all fresh 2.60 enrich

# # === Define core geometry dimensions ===
assembly_ij_dim = 21.5    # cm (length of one assembly side in X/Y)(8.466 in) https://www.nrc.gov/docs/ML2022/ML20224A492.pdf
fuel_k_dim = 200          # cm (height of fuel region) (78.74 in) https://www.nrc.gov/docs/ML2022/ML20224A492.pdf
bottom_ref_k_dim = 10.16  # cm (height of bottom reflector) (4.00 in) https://www.nrc.gov/docs/ML2022/ML20224A492.pdf
top_ref_k_dim = 9.02      # cm (height of top reflector) (3.551 in) https://www.nrc.gov/docs/ML2022/ML20224A492.pdf
thermal_power = 160       # MWt https://www.nrc.gov/docs/ML2022/ML20224A492.pdf

# # === Build matrices and solve for flux and k-effective ===
# A1, A2, B1_fast_fission, B1_thermal_fission, B2 = matrix(
#     core_2, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim
# )

# k_1, flux1_1, flux2_1 = fluxsearch(A1, A2, B1_fast_fission, B1_thermal_fission, B2)
# print(f"Final k-effective: {k_1:.6f}")

# with open("flux1_1.txt", 'w') as flux1_file:
#     for flux in flux1_1:
#         flux1_file.write(f"{flux}\n")
# with open("flux2_1.txt", 'w') as flux2_file:
#     for flux in flux2_1:
#         flux2_file.write(f"{flux}\n")

# === Run plot routine ===

# with open("flux1_1.txt", 'r') as flux1_file:
#     lines = flux1_file.readlines()
#     flux1_1 = np.array([float(line.strip()) for line in lines])
# with open("flux2_1.txt", 'r') as flux2_file:
#     lines = flux2_file.readlines()
#     flux2_1 = np.array([float(line.strip()) for line in lines])

# normalize_and_plot(
#     flux1_1, flux2_1, core_2,
#     power_MW=100,               # reactor thermal power
#     assembly_dim_cm=assembly_ij_dim,         # XY dimension per assembly
#     fuel_height_cm=200          # axial height of fuel
# )

# core_checker = CoreBuilder.core_maker("core_map_checker") # checkered 4.05 and 4.55

# A1_checker, A2_checker, B1_fast_fission_checker, B1_thermal_fission_checker, B2_checker = matrix(
#     grand_xs_library, core_checker, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim
# )

# k_checker, flux1_checker, flux2_checker = fluxsearch(
#     A1_checker, A2_checker, B1_fast_fission_checker, B1_thermal_fission_checker, B2_checker
# )
# print(f"Final k-effective: {k_checker:.6f}")
# with open("flux1_checker.txt", 'w') as flux1_file:
#     for flux in flux1_checker:
#         flux1_file.write(f"{flux}\n")
# with open("flux2_checker.txt", 'w') as flux2_file:
#     for flux in flux2_checker:
#         flux2_file.write(f"{flux}\n")

# # === Run plot routine ===
# with open("flux1_checker.txt", 'r') as flux1_file:
#     lines = flux1_file.readlines()
#     flux1_checker = np.array([float(line.strip()) for line in lines])
# with open("flux2_checker.txt", 'r') as flux2_file:
#     lines = flux2_file.readlines()
#     flux2_checker = np.array([float(line.strip()) for line in lines])

# normalize_and_plot(
#     flux1_checker, flux2_checker, core_checker,
#     power_MW=thermal_power,               # reactor thermal power
#     assembly_dim_cm=assembly_ij_dim,         # XY dimension per assembly
#     fuel_height_cm=fuel_k_dim          # axial height of fuel
# )

# core_NuScale_1800 = CoreBuilder.core_maker("core_map_NuScale_1800") # NuScale design at 1800 ppm

# A1_Nuscale_1800, A2_Nuscale_1800, B1_fast_fission_Nuscale_1800, B1_thermal_fission_Nuscale_1800, B2_Nuscale_1800 = matrix(
#     grand_xs_library, core_NuScale_1800, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim
#     )
# k_Nuscale_1800, flux1_Nuscale_1800, flux2_Nuscale_1800 = fluxsearch(
#     A1_Nuscale_1800, A2_Nuscale_1800, B1_fast_fission_Nuscale_1800, B1_thermal_fission_Nuscale_1800, B2_Nuscale_1800)
# print(f"Final k-effective: {k_Nuscale_1800:.6f}")
# A1_NuScale_eq, A2_NuScale_eq, B1_fast_fission_NuScale_eq, B1_thermal_fission_NuScale_eq, B2_NuScale_eq = matrix(
#     core_NuScale_eq, assembly_ij_dim, fuel_k_dim, bottom_ref_k_dim, top_ref_k_dim
# )

# k_NuScale_eq, flux1_NuScale_eq, flux2_NuScale_eq = fluxsearch(
#     A1_NuScale_eq, A2_NuScale_eq, B1_fast_fission_NuScale_eq, B1_thermal_fission_NuScale_eq, B2_NuScale_eq
# )
# print(f"Final k-effective: {k_NuScale_eq:.6f}")
# with open("flux1_NuScale_eq.txt", 'w') as flux1_file:
#     for flux in flux1_NuScale_eq:
#         flux1_file.write(f"{flux}\n")
# with open("flux2_NuScale_eq.txt", 'w') as flux2_file:
#     for flux in flux2_NuScale_eq:
#         flux2_file.write(f"{flux}\n")

# # === Run plot routine ===
# with open("flux1_NuScale_eq.txt", 'r') as flux1_file:
#     lines = flux1_file.readlines()
#     flux1_NuScale_eq = np.array([float(line.strip()) for line in lines])
# with open("flux2_NuScale_eq.txt", 'r') as flux2_file:
#     lines = flux2_file.readlines()
#     flux2_NuScale_eq = np.array([float(line.strip()) for line in lines])

# normalize_and_plot(
#     flux1_NuScale_eq, flux2_NuScale_eq, core_NuScale_eq,
#     power_MW=thermal_power,               # reactor thermal power
#     assembly_dim_cm=assembly_ij_dim,         # XY dimension per assembly
#     fuel_height_cm=fuel_k_dim          # axial height of fuel
# )

core_map_NuScale_eq = [ 
    ['rad', 0], ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],
    ['rad', 0], ['rad', 0],         ['rad', 0],         ['NUu405c00', 0],   ['NUu455c50', 17.5],['NUu405c00', 0],   ['rad', 0],         ['rad', 0],         ['rad', 0],
    ['rad', 0], ['rad', 0],         ['NUu455c50', 0],   ['NUu405c00', 17.5],['NUu405c00', 35],  ['NUu405c00', 17.5],['NUu455c50', 0],   ['rad', 0],         ['rad', 0],
    ['rad', 0], ['NUu405c00', 0],   ['NUu405c00', 17.5],['NUu455c50', 35],  ['NUu405c00', 35],  ['NUu455c50', 35],  ['NUu405c00', 17.5],['NUu405c00', 0],   ['rad', 0],
    ['rad', 0], ['NUu455c50', 17.5],['NUu405c00', 35],  ['NUu405c00', 35],  ['NUu260c00', 0],   ['NUu405c00', 35],  ['NUu405c00', 35],  ['NUu455c50', 17.5],['rad', 0],
    ['rad', 0], ['NUu405c00', 0],   ['NUu405c00', 17.5],['NUu455c50', 35],  ['NUu405c00', 35],  ['NUu455c50', 35],  ['NUu405c00', 17.5],['NUu405c00', 0],   ['rad', 0],
    ['rad', 0], ['rad', 0],         ['NUu455c50', 0],   ['NUu405c00', 17.5],['NUu405c00', 35],  ['NUu405c00', 17.5],['NUu455c50', 0],   ['rad', 0],         ['rad', 0],
    ['rad', 0], ['rad', 0],         ['rad', 0],         ['NUu405c00', 0],   ['NUu455c50', 17.5],['NUu405c00', 0],   ['rad', 0],         ['rad', 0],         ['rad', 0],
    ['rad', 0], ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],         ['rad', 0],
] 

core_map_checker = [
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],        ['rad', 0],
    ['rad', 0], ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],
    ['rad', 0], ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['rad', 0],
    ['rad', 0], ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],
    ['rad', 0], ['rad', 0],        ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['NUu405c00', 0],  ['NUu455c50', 0],  ['NUu405c00', 0],  ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],
]

# === Bisection to find critical boron concentration ===
# time_start = time.time()
# boron_critical, k_eff, flux1_crit, flux2_crit = bisection_boron(
#     core_name="core_map_NuScale_eq_crit_search",
#     assembly_map=core_map_NuScale_eq,
#     assembly_ij_dim=assembly_ij_dim,
#     fuel_k_dim=fuel_k_dim,
#     bottom_ref_k_dim=bottom_ref_k_dim,
#     top_ref_k_dim=top_ref_k_dim,
#     boron_low=700,
#     boron_high=800,
# )
# time_end = time.time()
# print(f"Time taken for crit search: {time_end - time_start:.2f} seconds")

# with open("flux1_crit.txt", 'w') as flux1_file:
#     for flux in flux1_crit:
#         flux1_file.write(f"{flux}\n")
# with open("flux2_crit.txt", 'w') as flux2_file:
#     for flux in flux2_crit:
#         flux2_file.write(f"{flux}\n")

# == Bisection to find critical boron concentration for checkered design ===
time_start = time.time()
boron_critical, k_eff, flux1_crit, flux2_crit = bisection_boron(
    core_name="core_map_checker_crit_search",
    assembly_map=core_map_checker,
    assembly_ij_dim=assembly_ij_dim,
    fuel_k_dim=fuel_k_dim,
    bottom_ref_k_dim=bottom_ref_k_dim,
    top_ref_k_dim=top_ref_k_dim,
)
time_end = time.time()
print(f"Time taken for crit search: {time_end - time_start:.2f} seconds")

with open("flux1_checker_crit.txt", 'w') as flux1_file:
    for flux in flux1_crit:
        flux1_file.write(f"{flux}\n")
with open("flux2_checker_crit.txt", 'w') as flux2_file:
    for flux in flux2_crit:
        flux2_file.write(f"{flux}\n")

        
# === Run plot routine ===
with open("flux1_checker_crit.txt", 'r') as flux1_file:
    lines = flux1_file.readlines()
    flux1_crit = np.array([float(line.strip()) for line in lines])
with open("flux2_checker_crit.txt", 'r') as flux2_file:
    lines = flux2_file.readlines()
    flux2_crit = np.array([float(line.strip()) for line in lines])

# interp_cx_for_graph = interpolate_xs(grand_xs_library, 758.984375)
# core_NuScale_crit = CoreBuilder.core_maker("core_map_NuScale_eq_crit_search")
core_checker_crit = CoreBuilder.core_maker("core_map_checker_crit_search")
normalize_and_plot(
    flux1_crit, flux2_crit, core_checker_crit,
    power_MW=thermal_power,               # reactor thermal power
    assembly_dim_cm=assembly_ij_dim,         # XY dimension per assembly
    fuel_height_cm=fuel_k_dim          # axial height of fuel
)