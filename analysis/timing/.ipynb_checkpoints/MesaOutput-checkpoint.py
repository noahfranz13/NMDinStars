# Helpful class for reading in MESA output files
class MesaOutput():

    def __init__(self, dirPath):
        self.dirPath = dirPath

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
        return glob.glob(os.path.join(self.dirPath, '*/*.out'))
