import os
import shutil

def delete_junk_files(base_dir='images'):
    junk_files = ['.DS_Store']
    deleted_files = 0
    deleted_dirs = 0

    for root, dirs, files in os.walk(base_dir, topdown=False):
        # Hapus file sampah
        for f in files:
            if f in junk_files or f.startswith('._'):
                file_path = os.path.join(root, f)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        
        # Hapus folder __MACOSX
        for d in dirs:
            if d == '__MACOSX':
                dir_path = os.path.join(root, d)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Failed to delete directory {dir_path}: {e}")

    print(f"\nFinished! Deleted {deleted_files} junk files and {deleted_dirs} junk directories.")

if __name__ == "__main__":
    delete_junk_files('images')
