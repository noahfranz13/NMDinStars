# Helpful class for reading in MESA output files
class MesaOutput():

    def __init__(self, dirPath, read=False):
        self.dirPath = dirPath
        self.dataPaths = self.getDataFiles()
        self.terminalPaths = self.getTerminalOut()
        if read:
            self.data = self.getData()

    def getDataFiles(self):
        '''
        Get the data output files from dirPath
        '''
        import os, glob
        return glob.glob(os.path.join(self.dirPath, '*/*.data'))

    def getTerminalOut(self):
        '''
        Get the terminal output files from dirPath
        '''
        import os, glob
        return glob.glob(os.path.join(self.dirPath, '*/*.txt'))

    def getData(self):
        '''
        Read in all the .data files using mesa_reader

        Returns : MesaData object of each file
        '''
        from mesa_reader import MesaData
        return [MesaData(f) for f in self.dataPaths]
        
    def onlyConverging(self):
        '''
        Seperates files into convering and those that  didn't converge

        Returns: non-converging data (sets self.data = converging data)
        '''

        import os
        import subprocess as sp
        import numpy as np
        
        #print(os.path.join(self.dirPath, '*/*'))
        grep = sp.run([f"grep -r 'terminated evolution: cannot find acceptable model' {os.path.join(self.dirPath, '*/*')}" ], stderr=sp.PIPE, stdout=sp.PIPE, text=True, shell=True)
        
        output = grep.stdout
                
        bad = [o.split(':')[0] for o in output.split('\n')]
        if len(bad) < 1:
            raise FileNotFoundError(grep.stderr)

        indexes = []
        tp = np.array(self.terminalPaths)
        
        for f in bad[:-1]:
            indexes.append(np.where(f == tp)[0][0])
    
        indexes = np.array(indexes)
        
        badData = np.array(self.dataPaths)[indexes]
        oppMask = np.ones(len(self.dataPaths), dtype=bool)
        oppMask[indexes] = 0
        self.dataPaths = np.array(self.dataPaths)[oppMask]
        #self.dataPaths = [m.file_name for m in self.data]
        
        return badData

    def checkTime(self):
        '''
        Looks for empty directories to see if any process
        ran out of time

        Return : true if there are models that ran out of time
                 false if all models finished in time
        '''
        import os, glob
        import pandas as pd

        # get full dataframe of grid
        gridFiles = glob.glob(os.path.join(os.getenv('HOME'), 'NMDinStars', 'makeGrids', '*.txt'))
        
        grid = []
        for gg in gridFiles:
             df = pd.read_csv(gg, header=None, index_col=0, sep='\t')
             grid.append(df)
             
        grid = pd.concat(grid)
                              
        dirs = glob.glob(os.path.join(self.dirPath, '*'))
        print(dirs)
        toRerun = []
        for dd in dirs:
            files = glob.glob(os.path.join(dd, '*'))
            print(files)
            if len(files) == 0:
                idx = int(dd.split('/')[-1].split('i')[-1])
                gridRow = grid.iloc[idx]
                toRerun.append(gridRow)

        if len(toRerun) > 0:
            outfile = os.path.join(os.getcwd(), 'rerun_grid.txt')
            toRerun = pd.concat(toRerun)

            print('Some models ran out of time...')
            for row in toRerun:
                print(row)
            print(f'Writing grid to {outfile}')
            
            toRerun.to_csv(outfile, header=None, sep='\t')
            return True
        else:
            print('None of the mdoels ran out of time!')

            return False
            
                
                
        
