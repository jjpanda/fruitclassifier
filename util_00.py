import os

#check directory exist
def checkdir(absolute_path):
    print('checking directory:', absolute_path)
    if not os.path.exists(absolute_path):
        print('Creating ''%s''' % absolute_path)
        os.makedirs(absolute_path)