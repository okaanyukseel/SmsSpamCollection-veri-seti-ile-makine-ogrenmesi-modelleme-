import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 1. Veriyi dosyadan oku
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({"ham": 0, "spam": 1})

# 2. Eğitim ve test ayır
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3. TF-IDF vektörleme
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Modeli eğit
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. Tahminler ve doğruluk
y_train_pred = model.predict(X_train_vec)
y_test_pred = model.predict(X_test_vec)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n📊 Sınıflandırma Raporu:\n")
print(classification_report(y_test, y_test_pred))

# 7. Kullanıcıdan SMS al ve tahmin yap
while True:
    sms = input("\nBir SMS gir (çıkmak için q): ")
    if sms.lower() == "q":
        break
    sms_vec = vectorizer.transform([sms])
    prediction = model.predict(sms_vec)[0]
    print("📩 Bu mesaj", "SPAM!" if prediction == 1 else "normal.")

# 6. Doğrulukları çiz
plt.bar(["Eğitim Doğruluğu", "Test Doğruluğu"], [train_acc, test_acc], color=["skyblue", "orange"])
plt.title("Model Doğruluğu")
plt.ylim(0.8, 1.0)
plt.ylabel("Doğruluk")
plt.grid(axis="y")
plt.show()


    


