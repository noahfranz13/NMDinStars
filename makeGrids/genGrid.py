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
    mu12 = np.logspace(-2, np.log10(4), n, endpoint=True) # mu_12 values

    # create grid
    if not useNMDM:
        grid = np.array([np.array([ii, jj, kk, mm, yy, zz]) for ii, mm in enumerate(m) for jj, yy in enumerate(y) for kk, zz in enumerate(z)])
    else:
        grid = np.array([np.array([ii, jj, kk, tt, mm, yy, zz, mu]) for ii, mm in enumerate(m) for jj, yy in enumerate(y) for kk, zz in enumerate(z) for tt, mu in enumerate(mu12)])

    # split grid based on even and odd indices
    gridDf = pd.DataFrame(grid, columns=['m_idx', 'y_idx', 'z_idx', 'u_idx', 'm', 'y', 'z', 'mu'])
    print(gridDf)
    

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
        np.savetxt(f'gridFile-{ii}.txt', gg, fmt=' '.join(['%i']*4 + ['%f']*4))

if __name__ == '__main__':
    sys.exit(main())

            
