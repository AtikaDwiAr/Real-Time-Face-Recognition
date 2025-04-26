# Real-Time Face Recognition

Project ini mengimplementasikan sistem Face Recognition secara real-time menggunakan metode Eigenfaces dengan OpenCV dan scikit-learn.

## Installation
Pastikan Python 3.8+ sudah terinstal. Install semua library yang diperlukan: 

Required packages:
- opencv-python
- scikit-learn
- matplotlib
- numpy

## Usage

### 1. Ekstrak Dataset
Run `extract_dataset.py` untuk mengekstrak file dataset gambar ke dalam folder `images/`:

```bash
python extract_dataset.py
```

Jika terdapat file sampah, run `delete_junk.py` untuk menghapusnya:

```bash
python delete_junk.py
```

### 2. Training Model
Run `main.py` untuk melatih model face recognition menggunakan dataset yang telah diekstraksi:

```bash
python main.py
```

Model yang terlatih akan disimpan dalam file `eigenface_pipeline.pkl`.

### 3. Real-Time Face Recognition
Untuk menjalankan face recognition secara real-time menggunakan webcam, run `main.py`:

```bash
python main.py
```

Webcam akan terbuka dan sistem akan mulai mengenali wajah secara langsung. Nama dan confidence level akan ditampilkan untuk setiap wajah yang dikenali.

Tekan `q` pada keyboard untuk keluar dari tampilan webcam.

## Project Structure
```plaintext
FDR_Project/
├── dataset.zip
├── eigenface_pipeline.pkl
├── delete_junk.py
├── extract_dataset.py
├── main.py
├── requirements.txt
└── README.md
```


