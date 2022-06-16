# Helpful class for reading in MESA output files
# imports
import os, sys, glob
import numpy as np
import pandas as pd

class MesaOutput():

    def __init__(self, dirPath, read=True):        
        self.dirPath = dirPath
        self.dataPaths = self.getDataFiles()
        self.terminalPaths = self.getTerminalOut()
        self.flags = np.zeros(len(self.dataPaths))
        if read:
            self.data = self.getData()
        else:
            self.data = None

    def getDataFiles(self):
        '''
        Get the data output files from dirPath
        '''
        return glob.glob(os.path.join(self.dirPath, '*/*.data'))

    def getTerminalOut(self):
        '''
        Get the terminal output files from dirPath
        '''
        return glob.glob(os.path.join(self.dirPath, '*/*.txt'))

    def getData(self):
        '''
        Read in all the .data files using mesa_reader

        Returns : MesaData object of each file
        '''
        from mesa_reader import MesaData
        return [MesaData(f) for f in self.dataPaths]
        
    def checkConverging(self):
        '''
        Seperates files into convering and those that  didn't converge

        Returns: non-converging data (sets self.data = converging data)
        '''
        
        import subprocess as sp

        grep = sp.run([f"grep -r 'termination code: power_he_burn_upper_limit' {os.path.join(self.dirPath, '*/*')}" ], stderr=sp.PIPE, stdout=sp.PIPE, text=True, shell=True)

        output = grep.stdout
        if len(grep.stderr) > 0:
            raise Exception(grep.stderr)
        
        good = [o.split(':')[0] for o in output.split('\n')]
        
        goodIdxs = []
        tp = np.array(self.terminalPaths)
        for f in good[:-1]: # the last element of stdout is always empty
            #print('*' + f)
            goodIdxs.append(np.where(f == tp)[0][0])
    
        indexes = np.array(goodIdxs)
        goodData = np.array(self.dataPaths)[indexes]
        oppMask = np.ones(len(self.dataPaths), dtype=bool)
        oppMask[indexes] = 0
        self.flags[oppMask] = 1 # give a flag of 1 without convergence
        print(f'WARNING : {len(np.where(self.flags == 1)[0])} models did not converge!')
        
    def checkTime(self):
        '''
        Looks for empty directories to see if any process
        ran out of time

        Return : false if there are models that ran out of time
                 true if all models finished in time
        '''
        
        # get full dataframe of grid
        gridFiles = glob.glob(os.path.join(os.getenv('HOME'), 'NMDinStars', 'makeGrids', '*.txt'))
        
        grid = []
        for gg in gridFiles:
             df = pd.read_csv(gg, header=None, index_col=0, sep='\t')
             grid.append(df)
             
        grid = pd.concat(grid)
                              
        dirs = glob.glob(os.path.join(self.dirPath, '*'))
        toRerun = []
        for dd in dirs:
            files = glob.glob(os.path.join(dd, '*'))
            if len(files) == 0:
                idx = int(dd.split('/')[-1].split('i')[-1])
                gridRow = grid.iloc[idx]
                toRerun.append(gridRow)

        if len(toRerun) > 0:
            outfile = os.path.join(os.getcwd(), 'rerun_grid.txt')
            toRerun = pd.concat(toRerun)

            print('Some models ran out of time...')
            print(toRerun)
            print(f'Writing grid to {outfile}')
            
            toRerun.to_csv(outfile, header=None, sep='\t')
            return False
        else:
            print('None of the models ran out of time!')

            return True

    def checkAge(self):
        '''
        checks the age of the outputs and flags anything 
        older than 13.77 byo with 3
        '''

        age = np.array([max(d.star_age) for d in self.data])
        whereOld = np.where((age > 13.77e9) * (self.flags == 0))[0]
        print(whereOld)
        print(age)
        self.flags[whereOld] = 3
        print(f'WARNING : {len(whereOld)} models are being flagged for age > 13.77 byo')


    def checkFlash(self):
        '''
        Check for unexpected shell flashing
        '''                

        
        
