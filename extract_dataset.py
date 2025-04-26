import zipfile
import os

# Nama file zip
zip_file = 'dataset.zip'
# Folder tujuan ekstraksi
extract_to = 'images'

# Cek apakah file zip ada
if os.path.exists(zip_file):
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed! Files are in '{extract_to}/' folder.")
else:
    print(f"Error: {zip_file} not found!")
