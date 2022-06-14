# Helpful class for reading in MESA output files
class MesaOutput():

    def __init__(self, dirPath, read=False):
        self.dirPath = dirPath
        self.dataPaths = self.getDataFiles()
        self.terminalPaths = self.getTerminalOut()
        if read:
            self.data = self.getData(n)

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

    def getData(self, numRead):
        '''
        Read in all the .data files using mesa_reader

        Returns : MesaData object of each file
        '''
        from mesa_reader import MesaData
        if not numRead:
            return [MesaData(f) for f in self.dataPaths]
        else:
            return [MesaData(self.dataPaths[ii]) for ii in range(numRead)]
        
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
