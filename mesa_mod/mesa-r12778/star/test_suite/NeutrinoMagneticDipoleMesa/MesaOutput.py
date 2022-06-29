# Helpful class for reading in MESA output files
# imports
import os, sys, glob
import numpy as np
import pandas as pd
from mesa_reader import MesaData
from multiprocessing import Pool
        

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

        self.index = self.getIndex()

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

    def getIndex(self):
        '''
        Get the index from the filepath
        '''
        dirs = glob.glob(os.path.join(self.dirPath, '*'))
        return np.array([dd.split('i')[-1] for dd in dirs])  

    def getDataHelper(self, f): return MesaData(f)

    def getData(self):
        '''
        Read in all the .data files using mesa_reader

        Returns : MesaData object of each file
        '''
        print('Reading in the data, this may take a while...')

        with Pool() as p:
            result = p.map(self.getDataHelper, self.dataPaths)
        
        return result

    def checkConverging(self):
        '''
        Seperates files into convering and those that  didn't converge

        Returns: non-converging data (sets self.data = converging data)
        '''
        
        import subprocess as sp

        cmd = [f"grep -r 'termination code: power_he_burn_upper_limit' {self.dirPath}" ]
        
        grep = sp.run(cmd, stderr=sp.PIPE, stdout=sp.PIPE, text=True, shell=True)

        output = grep.stdout
        if len(grep.stderr) > 0:
            raise Exception(grep.stderr)
        
        good = [o.split(':')[0] for o in output.split('\n')]
        
        goodIdxs = []
        tp = np.array(self.terminalPaths)
        print(f'Length of grep: {len(good)-1}\nLength of terminal paths: {len(tp)}')
        #print(tp)
        #print(good)
        for f in good[:-1]: # the last element of stdout is always empty
            idx = np.where(f == tp)[0]
            if len(idx > 0):
                goodIdxs.append(idx[0])
            else:
                print("No file found for: ", f)
            
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
        print(toRerun)
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
        self.flags[whereOld] = 3
        print(f'WARNING : {len(whereOld)} models are being flagged for age > 13.77 byo')


    def checkFlash(self):
        '''
        Check for unexpected shell flashing
        '''                
        
        goodIdxs = np.where(self.flags == 0)[0]
        
        for i, m in zip(goodIdxs, np.array(self.data)[goodIdxs]):
            
            he4 = np.array(m.center_he4)

            where1 = np.where(he4 > 0.95)[0]
            if len(where1) == 0:
                print('WARNING: Model never has He4=1, skipping check for this model!')
                continue
            
            lastIdx1 = where1[-1]
            
            whereLess1 = np.where(he4[lastIdx1:] < 0.9)[0]
            if len(whereLess1) > 0:
                self.flags[i] = 2
                                
        print(f'WARNING : {len(np.where(self.flags == 2)[0])} models He Flashed')
        
    def MD_findShellFlash(self):
        """
        From Mitchell Dennis, to be used with old data that doesn't have output center_he4

        This function determines whether or not a given MESA model is shell flashing.

        INPUTS:
        The power_he_burn MESA output.

        RETURNS:
        A Boolean that evaluates to True if a model is shell flashing
        """
        goodIdxs = np.where(self.flags == 0)[0]

        for ii, m in zip(goodIdxs, np.array(self.data)[goodIdxs]):
            heBurn = m.power_he_burn

            shellFlash = False
            heIgnitionFlag = False
            previousLuminosity = 0
            maxHeBurn = 0

            for i in range(len(heBurn)):
                if heBurn[i] > maxHeBurn:
                    maxHeBurn = heBurn[i]
                if (not heIgnitionFlag and heBurn[i] > 10**-6):
                    heIgnitionFlag = True
                if (not shellFlash and heIgnitionFlag and heBurn[i] < maxHeBurn / 5):
                    shellFlashIndex = i
                    shellFlash = True
                if (shellFlash and i > shellFlashIndex + 50):
                    #return True
                    self.flags[ii] == 2
