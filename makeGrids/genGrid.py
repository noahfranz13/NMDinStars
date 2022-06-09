'''
Script to compute a grid given a sidelength and
if mu_12 should be considered.
'''
# imports
import  numpy as np
import sys

def computeGrids(n, useNMDM):
    '''
    n [int] : edge length of grid
    useNMDM [bool] : true if using mu_12, false otherwise
    '''
    
    # first generate arrays of important info
    m = np.linspace(0.7, 2.25, n, endpoint=True) # mass
    y = np.linspace(0.2, 0.3, n, endpoint=True) # helium mass frac
    z = np.logspace(-5, -1.39794000867, n, endpoint=True) # metallicity
    mu12 = np.logspace(np.log10(5), -2, n, endpoint=True) # mu_12 values

    # create grid
    if not useNMDM:
        grid = np.array([np.array([mm, yy, zz]) for mm in m for yy in y for zz in z])
    else:
        grid = np.array([np.array([mm, yy, zz, mu]) for mm in m for yy in y for zz in z for mu in mu12])

    # check length and if >25,000 write to separate grids
    if len(grid) > 25000:
        grids = []
        for i in range(0, len(grid), 25000):
            try:
                grids.append(grid[i:i+25000])
            except:
                grids.append(grid[i:])
    else:
        grids = [grid]

    return grids

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help="side length of grid", type=int, default=None)
    parser.add_argument('--no-NMDM', dest='useNMDM', action='store_false')
    parser.set_defaults(useNMDM=True)
    args = parser.parse_args()
    grids = computeGrids(args.n, args.useNMDM)

    # write file(s) with these grids
    for ii, gg in enumerate(grids):
        np.savetxt(f'gridFile-{ii}.txt', gg)

if __name__ == '__main__':
    sys.exit(main())

            
