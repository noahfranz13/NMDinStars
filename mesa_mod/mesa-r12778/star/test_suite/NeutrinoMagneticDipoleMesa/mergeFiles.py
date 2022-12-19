# Short script to combine the following files based on index
# 1) input grid
# 2) iBandOutput.txt
# 3) colorCorrections.txt

import sys, os, glob
import pandas as pd

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="mesa output file", default='/media/ubuntu/T7/mesa-1296')
    args = parser.parse_args()

    if args.dir[-1] == '/':
        key = args.dir[:-1].split('/')[-1]
    else:
        key = args.dir.split('/')[-1]
    
    color = pd.read_csv('colorCorrections.txt', header=None, index_col=0)
    color.columns = ['our_flag', 'worthey_lee_flag', 'M_I', 'M_I_err', 'V_I', 'V_I_err']
    outData = pd.read_csv('iBandOutput.txt', sep='\t', index_col=0, header=None)
    outData.columns = ['flag', 'surface_grav', 'Teff', 'feh', 'L']

    onMana = False
    if onMana:
        gridFiles = glob.glob(os.path.join(os.getenv('HOME'), 'NMDinStars', 'makeGrids', f'*.txt'))
    else:
        gridFiles = glob.glob(os.path.join(os.getenv('HOME'), 'Documents', 'NMDinStars', 'makeGrids', f'*.txt'))

    grid = []
    for gg in gridFiles:
        df = pd.read_csv(gg, header=None, index_col=0, sep='\t')
        grid.append(df)
    grid = pd.concat(grid)
    grid.columns = ['mass_index', 'y_index', 'z_index', 'mu_index', 'mass', 'y', 'z', 'mu']
    
    allData = pd.concat([grid, outData, color], axis=1)
    allData.drop(columns='our_flag', inplace=True)
    allData.query('M_I == M_I', inplace=True)
    allData.to_csv(f'postProcess_output_{key}.txt')
    
if __name__ == '__main__':
    sys.exit(main())
