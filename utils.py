import os
import glob

import csv
from pathlib import Path

def convert_csv_to_relative(csv_path, base_prefix):
    csv_path = Path(csv_path)
    new_lines = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        new_lines.append(header)
        for row in reader:
            # Se asume que la ruta está en la primera columna.
            absolute_path = Path(row[0])
            # Convertir a cadena y quitar el prefijo absoluto.
            relative_str = str(absolute_path).replace(base_prefix, "")
            # Si sobra un separador inicial, quitarlo.
            if relative_str.startswith("/") or relative_str.startswith("\\"):
                relative_str = relative_str[1:]
            new_lines.append([relative_str])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_lines)
    print(f"Convertido: {csv_path}")

if __name__ == "__main__":
    # Define el prefijo absoluto que deseas eliminar.
    base_prefix = "/mnt/e/ISPRS-Potsdam-adri/"
    # Lista de archivos CSV a procesar.
    csv_files = [
        "/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/images.csv",
        "/mnt/e/ISPRS-Potsdam-adri/split_postdam_ir_512/train/labels.csv",
        # Puedes agregar otros CSV (por ejemplo, de test o validación) aquí.
    ]
    for csv_file in csv_files:
        convert_csv_to_relative(csv_file, base_prefix)


        
