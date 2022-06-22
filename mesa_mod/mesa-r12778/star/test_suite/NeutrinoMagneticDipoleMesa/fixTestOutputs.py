import os, glob

datafiles = glob.glob('/home/nfranz/lus_scratch/mesa-1296/*/*.data')
textfiles = glob.glob('/home/nfranz/lus_scratch/mesa-1296/*/*.txt')


for i, f in enumerate(datafiles):

    fOld = f
    ext = f.split('.')[-1]
    f = f.split('.')[0]
    '''fNew = f[:-7] + 'M' + f[-7:]
    fNew = fNew[:-5] + 'Y' + fNew[-5:]
    fNew = fNew[:-3] + 'Z' + fNew[-3:]
    fNew = fNew[:-1] + 'U' + fNew[-1]
    fNew += f'_i{i}'
    fNew += '.' + ext'''
    idx = f.split('/')[-2].split('i')[-1]
    fSplit = f.split('/')
    fNew = fSplit[-1].split('i')[0] + 'i' + idx + '.' + ext
    f1 = "/"
    for item in fSplit[:-1]:
        f1 = os.path.join(f1, item)
    fNew = os.path.join(f1, fNew)
    print(fOld, fNew)

    os.rename(fOld, fNew)

for i, f in enumerate(textfiles):

    fOld = f
    ext = f.split('.')[-1]
    f = f.split('.')[0]
    '''fNew = f[:-7] + 'M' + f[-7:]
    fNew = fNew[:-5] + 'Y' + fNew[-5:]
    fNew = fNew[:-3] + 'Z' + fNew[-3:]
    fNew = fNew[:-1] + 'U' + fNew[-1]
    fNew += f'_i{i}'
    fNew += '.' + ext'''
    
    idx = f.split('/')[-2].split('i')[-1]
    fSplit = f.split('/')
    fNew = fSplit[-1].split('i')[0] + 'i' + fSplit[-1].split('i')[1] + 'i' + idx + '.' + ext
    #print(fNew)
    f1 = "/"
    for item in fSplit[:-1]:
        f1 = os.path.join(f1, item)
    fNew = os.path.join(f1, fNew)
    print(fOld, fNew)

    os.rename(fOld, fNew)
