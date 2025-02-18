import os
import glob

# Ruta base donde se encuentran los CSV (train, test, validation)
base_dir = '/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512'

# Prefijos a reemplazar
new_prefix = '/mnt/e/ISPRS-Potsdam-adri/postdam_ir_512'

# Directorios a procesar
subdirs = ['train', 'test', 'validation']

for subdir in subdirs:
    folder = os.path.join(base_dir, subdir)
    # Procesamos todos los archivos CSV en la carpeta
    for csv_file in glob.glob(os.path.join(folder, '*.csv')):
        # Leemos el contenido del CSV
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        # Reemplazamos el prefijo en cada línea
        new_lines = [line.replace(old_prefix, new_prefix) for line in lines]
        # Escribimos los cambios de vuelta al archivo
        with open(csv_file, 'w') as f:
            f.writelines(new_lines)
        print(f'Procesado: {csv_file}')
