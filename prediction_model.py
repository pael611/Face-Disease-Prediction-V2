# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: main-ds
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import glob


# %%
dataset_path = 'dataset'
categories = ['Acne', 'Blackheads', 'Dark Spots', 'Normal Skin', 'Oily Skin', 'Wrinkles']
img_size = 224
BATCH_SIZE = 64
num_classes = len(categories)

# %%
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# %%
print("Mapping folder ke label (class_indices):", train_gen.class_indices)
print("Urutan categories:", categories)
print("Distribusi label di data training:", np.bincount(train_gen.classes))

# %%
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# %%
# 15. Siapkan daftar file & label, split stratified
import os, glob
from sklearn.model_selection import train_test_split

# Gunakan variabel yang sudah ada: dataset_path, categories, img_size, BATCH_SIZE
# Hanya gunakan format yang didukung decoder TensorFlow: JPEG, PNG, BMP, GIF
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")

filepaths = []
labels = []
for cls_idx, cls_name in enumerate(categories):
    cls_dir = os.path.join(dataset_path, cls_name)
    cls_files = []
    for ext in image_exts:
        cls_files.extend(glob.glob(os.path.join(cls_dir, ext)))
    # Tambahkan ke list global
    filepaths += cls_files
    labels += [cls_idx] * len(cls_files)

# Filter defensif: pastikan ekstensi valid (hindari file non-gambar)
valid_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
filtered = [(fp, lab) for fp, lab in zip(filepaths, labels) if fp.lower().endswith(valid_suffix)]
filepaths = [fp for fp, _ in filtered]
labels = [lab for _, lab in filtered]

# Split stratified untuk train/valid (20% validasi)
X_train, X_val, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Tampilkan ringkas distribusi sebelum balancing
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print("Distribusi train sebelum balancing:")
for u, c in zip(unique, counts):
    print(f"  {categories[u]}: {c}")

# %%
# 16. Bangun tf.data untuk oversampling per-kelas (robust decoding via PIL)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

# Decoder robust menggunakan PIL agar mendukung lebih banyak format dan menghindari crash
def pil_decode_and_resize(path):
    path_str = path.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        arr = np.asarray(img, dtype=np.float32)
        return arr

def _load_and_preprocess(path, label):
    # Gunakan tf.py_function untuk memanggil PIL
    img = tf.py_function(func=pil_decode_and_resize, inp=[path], Tout=tf.float32)
    # Set shape statis agar Keras tahu dimensi
    img = tf.reshape(img, [img_size, img_size, 3])
    # MobileNetV2 preprocessing
    img = preprocess_input(img)
    return img, label

# Dataset validasi (tanpa oversampling)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Dataset per-kelas untuk train
per_class_datasets = []
class_counts = []
for cls_idx, cls_name in enumerate(categories):
    cls_files = [fp for fp, lab in zip(X_train, y_train) if lab == cls_idx]
    class_counts.append(len(cls_files))
    ds = tf.data.Dataset.from_tensor_slices((cls_files, [cls_idx] * len(cls_files)))
    ds = ds.shuffle(max(8*BATCH_SIZE, len(cls_files)))
    ds = ds.repeat()  # penting agar sampler bisa menarik elemen terus-menerus
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    per_class_datasets.append(ds)

# Bobot sampling: seimbangkan kelas (semua sama besar)
num_classes = len(categories)
weights = [1.0/num_classes] * num_classes

balanced_ds = tf.data.experimental.sample_from_datasets(per_class_datasets, weights=weights)
train_ds_balanced = balanced_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Oversampling aktif. Batch train akan berisi distribusi kelas ~seimbang. Decoder: PIL.")

# %% [markdown]
# <h1>Membangun Model</h1>

# %%
#Transfer Learning Model (MobileNetV2 + Decision Layer)
base_model = MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# %%
inputs = Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# %%
# 7. Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    'best_skin_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.summary()

# %%
# 17. Latih model menggunakan dataset seimbang (tanpa class_weight)
# Pastikan model, callbacks (early_stopping, checkpoint, reduce_lr) sudah didefinisikan sebelumnya.
# Tentukan steps_per_epoch (jumlah batch per epoch) karena train_ds_balanced.repeat() bersifat tak terbatas.

steps_per_epoch = max(1, sum(class_counts) // BATCH_SIZE)
print("steps_per_epoch:", steps_per_epoch)

history_balanced = model.fit(
    train_ds_balanced,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Plot training curves (accuracy & loss) for balanced training
def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history_balanced)

# %%
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_condition(model, img_path, categories):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, categories[predicted_class], confidence

def show_recommendations(predicted_label):
    df = pd.read_csv('skincare_product/treatment.csv')
    produk = df[df['Tags'].str.lower() == predicted_label.lower()]
    if produk.empty:
        print("Tidak ada rekomendasi produk untuk kategori ini.")
        return
    print(f"\nRekomendasi produk untuk '{predicted_label}':")
    for _, row in produk.iterrows():
        print(f"- {row['Brand']} | {row['Product Name']} | {row['Price']}")
        print(f"  Link: {row['Links']}")
        # Cek semua kemungkinan ekstensi gambar
        img_found = False
        for ext in ['png', 'jpg', 'jpeg', 'webp']:
            img_path = f"skincare_product/gambar_produk/{row['Id']}.{ext}"
            if os.path.exists(img_path):
                img_prod = image.load_img(img_path, target_size=(720, 1280))
                plt.figure()
                plt.imshow(img_prod)
                plt.title(f"{row['Brand']} - {row['Product Name']}")
                plt.axis('off')
                plt.show()
                img_found = True
                break
        if not img_found:
            # Jika tidak ditemukan, coba cari dengan glob (jaga-jaga ada nama file aneh)
            pattern = f"skincare_product/gambar_produk/{row['Id']}.*"
            matches = glob.glob(pattern)
            if matches:
                img_prod = image.load_img(matches[0], target_size=(720, 1280))
                plt.figure()
                plt.imshow(img_prod)
                plt.title(f"{row['Brand']} - {row['Product Name']}")
                plt.axis('off')
                plt.show()
            else:
                print(f"  Gambar produk tidak ditemukan untuk ID: {row['Id']}")

# Contoh penggunaan prediksi gambar baru
img_path = "darkspot.jpg"  # Ganti dengan path gambar yang ingin diuji
if os.path.exists(img_path):
    predicted_class_index, predicted_class_name, confidence = predict_skin_condition(model, img_path, categories)
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class name: {predicted_class_name}")
    print(f"Confidence: {confidence:.2f}%")
    img_disp = image.load_img(img_path, target_size=(720, 1280))
    plt.imshow(img_disp)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()
    show_recommendations(predicted_class_name)
else:
    print(f"File {img_path} tidak ditemukan.")

# %%
# 18. Evaluasi: Confusion Matrix & Classification Report pada validation set
print("\nEvaluasi pada validation set (confusion matrix & classification report)...")

# Kumpulkan prediksi pada val_ds
y_true = np.array(y_val)
y_pred = []
for batch_imgs, _ in val_ds:
    probs = model.predict(batch_imgs, verbose=0)
    batch_pred = np.argmax(probs, axis=1)
    y_pred.extend(batch_pred)
y_pred = np.array(y_pred[:len(y_true)])  # jaga-jaga jika overshoot

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Confusion Matrix:\n", cm)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Validation)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix disimpan ke 'confusion_matrix.png'.")

# Classification report
report = classification_report(y_true, y_pred, target_names=categories, digits=4)
print("\nClassification Report:\n", report)
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("Classification report disimpan ke 'classification_report.txt'.")

# %%
# 13. SIMPAN MODEL
model.save('best_skin_model.h5')

# %%
#14.EKSPOR MODEL KE ONNX
import tf2onnx
import onnx
model = tf.keras.models.load_model('best_skin_model.h5')
spec = (tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="input_1"),)
output_path = 'best_skin_model.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
onnx.save_model(model_proto, output_path)
print(f"Model berhasil diekspor ke {output_path}")

# %%
