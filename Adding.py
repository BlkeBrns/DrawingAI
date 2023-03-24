import os
import glob

filenames = glob.glob('image*.png')

for file in filenames:
    diff = file.find('.') - file.find('_')
    if diff < 4:
        thing = 4 - diff
        if thing == 2:
            os.rename(file, file[0:file.find('_')+1] + "00" + file[file.find("_") + 1:])
        elif thing == 1:
            os.rename(file, file[0:file.find('_')+1] + "0" + file[file.find("_") + 1:])
