import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CARI KUMPULAN DATASET (DARI GITHUB GIST) ---
url = 'https://gist.githubusercontent.com/agtbaskara/a1a7017027cc1df9d35cf06e1e5575b7/raw/dataset_sms_spam_v2.csv'
df = pd.read_csv(url)

# Menyesuaikan nama kolom jika diperlukan (biasanya 'Teks' dan 'label')
# Kita pastikan datanya lebih dari 40 sesuai syarat tugas
print(f"Total data yang ditemukan: {len(df)} baris")

# --- 2. PEMBAGIAN DATA (80% Training : 20% Testing) ---
X = df['Teks']  # Kolom teks SMS
y = df['label'] # Kolom kategori (0 untuk ham, 1 untuk spam)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- 3. LAKUKAN EKSTRAKSI FITUR (TF-IDF) ---
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- 4. LAKUKAN TRAINING DATA (Naive Bayes) ---
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# --- 5. LAKUKAN TESTING KLASIFIKASI ---
y_pred = model.predict(X_test_tfidf)

# --- 6. AKURASI ---
acc = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"Hasil Akurasi Model: {acc * 100:.2f}%")
print("-" * 30)
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Uji coba dengan input baru
sms_baru = ["Selamat Anda mendapatkan hadiah 10jt, klik link ini", "Lagi dimana? Besok jadi kumpul?"]
prediksi = model.predict(vectorizer.transform(sms_baru))
for s, p in zip(sms_baru, prediksi):
    label = "Spam" if p == 1 else "Bukan Spam (Ham)"
    print(f"SMS: '{s}' -> Prediksi: {label}")