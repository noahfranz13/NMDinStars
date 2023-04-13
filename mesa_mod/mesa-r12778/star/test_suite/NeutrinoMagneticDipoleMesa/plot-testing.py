# plot up the testing data

from MesaOutput import MesaOutput

def main():

    dirpath = '/home/nfranz/lus_scratch/mesa-testing'

    m = MesaOutput(dirpath)

    print(m)

if __name__ == '__main__':
    main()
