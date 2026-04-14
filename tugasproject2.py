import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATASET DARI GITHUB ---
url = 'https://raw.githubusercontent.com/gevabriel/dataset/main/indo_spam.csv'

try:
    df = pd.read_csv(url)
except:
    df = pd.read_csv(url, encoding='latin-1')

# --- 2. PRE-PROCESSING & CLEANING ---
# Menentukan kolom X (teks) dan y (label)
if 'label' in df.columns and 'pesan' in df.columns:
    X = df['pesan']
    y = df['label']
else:
    X = df.iloc[:, 1]
    y = df.iloc[:, 0]

# --- 3. TRAINING MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Menggunakan Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Cek Akurasi Model
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Siap! Akurasi Data Uji: {acc * 100:.2f}%")
print("Ketik 'exit' untuk berhenti.")
print("-" * 50)

# --- 4. INTERACTIVE TESTING DENGAN TINGKAT KEYAKINAN ---
while True:
    user_input = input("\nMasukkan pesan SMS untuk dianalisa: ")
    
    if user_input.lower() == 'exit':
        print("Program dihentikan. Selamat mengerjakan tugas!")
        break
    
    # Transformasi input ke format TF-IDF
    input_tfidf = vectorizer.transform([user_input])
    
    # Prediksi Label (Spam/Ham)
    prediksi = model.predict(input_tfidf)[0]
    
    # Prediksi Probabilitas (Tingkat Keyakinan)
    # predict_proba mengembalikan list peluang tiap kelas [P(Ham), P(Spam)]
    probabilitas = model.predict_proba(input_tfidf)[0]
    
    # Ambil nilai tertinggi sebagai tingkat keyakinan
    confidence = max(probabilitas) * 100
    
    # Penentuan warna/status visual sederhana
    if prediksi.lower() == 'spam':
        icon = "🚩 [SPAM]"
    else:
        icon = "✅ [BUKAN SPAM]"
        
    print(f"Hasil Analisa : {icon}")
    print(f"Tingkat Keyakinan : {confidence:.2f}%")
    print("-" * 50)