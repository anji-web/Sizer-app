# Sizer App - Aplikasi Pengukuran Tubuh

Python version : 3.11.9

## Fitur

- **Pengukuran Otomatis**: Mengukur berbagai dimensi tubuh menggunakan MediaPipe Pose
- **Validasi Pose**: Memastikan pose yang benar sebelum pengukuran
- **Penyimpanan Gambar**: Otomatis menyimpan gambar hasil capture
- **Export Excel**: Hasil pengukuran diekspor ke file Excel
- **Akurasi Report**: Menampilkan perbandingan dengan ukuran manual

## Fitur Penyimpanan Gambar

Setelah pose valid dan pengukuran selesai, aplikasi akan:
1. Otomatis menyimpan gambar hasil capture ke folder `captured_images/`
2. Format nama file: `NRP_Nama_YYYYMMDD_HHMMSS.jpg`
3. Menampilkan notifikasi popup dengan lokasi file
4. Gambar disimpan dengan pose landmarks yang sudah terdeteksi

### Contoh Nama File
```
123456_John_Doe_20231111_143025.jpg
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
python main.py
```

## Struktur Folder

```
Sizer-app/
├── main.py                 # File utama aplikasi
├── requirements.txt        # Dependencies
├── buildozer.spec         # Config untuk build Android
├── assets/                # Folder assets (logo, dll)
├── captured_images/       # Folder hasil capture (auto-generated)
└── hasil_ukur_*.xlsx      # File Excel hasil pengukuran
```

## Catatan

- Folder `captured_images/` akan dibuat otomatis saat pertama kali menyimpan gambar
- Gambar disimpan dalam format JPG dengan kualitas maksimal
- Nama file menggunakan timestamp untuk menghindari duplikasi
