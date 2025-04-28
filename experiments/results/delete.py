import os
import shutil

# Ścieżka do katalogu głównego
root_dir = 'exp_1_reg'

# Przechodzimy przez wszystkie katalogi i podkatalogi
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'algorithm3' in dirnames:
        folder_to_delete = os.path.join(dirpath, 'algorithm3')
        print(f"Usuwam folder: {folder_to_delete}")
        shutil.rmtree(folder_to_delete)