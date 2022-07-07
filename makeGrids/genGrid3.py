'''
Script to compute a grid given a sidelength and
if mu_12 should be considered.
'''
# imports
import pandas as pd
import  numpy as np
import sys

def computeGrids(useNMDM):
    '''
    n [int] : edge length of grid
    useNMDM [bool] : true if using mu_12, false otherwise
    '''
    
    # first generate arrays of important info
    m = np.linspace(0.7, 2.25, 15, endpoint=True) # mass
    y = np.linspace(0.2, 0.3, 10, endpoint=True) # helium mass frac
    z = np.logspace(-5, -1.39794000867, 25, endpoint=True) # metallicity
    mu12 = np.linspace(4.25, 6, 8, endpoint=True) # mu_12 values
    print(mu12)
    # create grid
    if not useNMDM:
        grid = np.array([np.array([ii, jj, kk, mm, yy, zz]) for ii, mm in enumerate(m) for jj, yy in enumerate(y) for kk, zz in enumerate(z)])
    else:
        grid = np.array([np.array([ii, jj, kk, tt, mm, yy, zz, mu]) for ii, mm in enumerate(m) for jj, yy in enumerate(y) for kk, zz in enumerate(z) for tt, mu in enumerate(mu12)])

    # split grid based on even and odd indices
    gridDf = pd.DataFrame(grid, columns=['m_idx', 'y_idx', 'z_idx', 'u_idx', 'm', 'y', 'z', 'mu'])

    grid = gridDf.astype({'m_idx': int, 'y_idx': int, 'z_idx': int, 'u_idx': int})
    
    # check length and if >25,000 write to separate grids
    
    if len(grid) > 25000:
        g = []
        for i in range(0, len(grid), 25000):
            try:
                g.append(grid[i:i+25000])
            except:
                g.append(grid[i:])
    else:
        g = [grid]

    print(g)
    return g

def main():

    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--n', help="side length of grid", type=int, default=None)
    parser.add_argument('--no-NMDM', dest='useNMDM', action='store_false')
    parser.set_defaults(useNMDM=True)
    args = parser.parse_args()
    grids = computeGrids(args.useNMDM)
    
    for ii, gg in enumerate(grids):
        gg.to_csv(f'third-gridFile-{ii}.txt', header=None, sep='\t', float_format='%.15f')
    
if __name__ == '__main__':
    sys.exit(main())

            
