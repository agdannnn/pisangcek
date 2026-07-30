import os

# Lokasi dataset
dataset_path = "dataset"

# Nama folder dataset
folders = ["train", "validation", "test"]

print("=" * 50)
print("PENGECEKAN DATASET")
print("=" * 50)

total_semua = 0

for folder in folders:
    print(f"\n📂 {folder.upper()}")

    folder_path = os.path.join(dataset_path, folder)

    total_folder = 0

    # Ambil semua nama kelas
    classes = sorted(os.listdir(folder_path))

    for kelas in classes:
        kelas_path = os.path.join(folder_path, kelas)

        if os.path.isdir(kelas_path):
            jumlah = len([
                file for file in os.listdir(kelas_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            print(f"{kelas:<20}: {jumlah} gambar")

            total_folder += jumlah

    print(f"Total {folder:<12}: {total_folder} gambar")

    total_semua += total_folder

print("\n" + "=" * 50)
print(f"TOTAL DATASET : {total_semua} gambar")
print("=" * 50)