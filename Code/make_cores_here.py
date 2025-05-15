from core_builder import CoreBuilder

###
# Use this file to create the core maps and turn them into the config files
# core maps should follow the format of "core_map2" below where it is 9by9
# and the first element is the assembly ID (name of cross section segment library. check the top of NE_571_project_5_main), 
# the second is the burnup to allow for selecting once and twice burned assemblies as desired (15 for once, 30 for twice).

### add core maps below
core_map2 = [
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],        ['rad', 0],
    ['rad', 0], ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],
    ['rad', 0], ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],
    ['rad', 0], ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],
    ['rad', 0], ['rad', 0],        ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['NUu260c00', 0],  ['NUu260c00', 0],  ['NUu260c00', 0],  ['rad', 0],        ['rad', 0],        ['rad', 0],
    ['rad', 0], ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],        ['rad', 0],
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

assembly_only = [
    ['NUu260c00', 0]
]

### run config_maker below. Comment out when done to avoid overwriting the config files

CoreBuilder.config_maker("core_map_checker_1800", core_map_checker, 1019.3, 557, 1800)  



