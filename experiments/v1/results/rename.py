import os

# Ścieżka do katalogu głównego
root_dir = 'exp_1_reg'

# Przechodzimy przez wszystkie katalogi i podkatalogi
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'algorithm4' in dirnames:
        old_path = os.path.join(dirpath, 'algorithm4')
        new_path = os.path.join(dirpath, 'algorithm')
        print(f"Zmiana nazwy: {old_path} -> {new_path}")
        os.rename(old_path, new_path)