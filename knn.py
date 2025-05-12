import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Veriyi oku
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({"ham": 0, "spam": 1})

# 2. Eğitim-test bölmesi
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3. TF-IDF vektörleme
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=5)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. KNN modeli
model = KNeighborsClassifier(n_neighbors=5)  # k=5 komşu
model.fit(X_train_vec, y_train)

# 5. Tahmin ve doğruluk
y_train_pred = model.predict(X_train_vec)
y_test_pred = model.predict(X_test_vec)
y_test_proba = model.predict_proba(X_test_vec)[:, 1]  # ROC için

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# 6. Sınıflandırma Raporu
print("\n📊 Sınıflandırma Raporu:\n")
print(classification_report(y_test, y_test_pred))

# 7. Eğitim/Test Doğruluk Grafiği
plt.figure(figsize=(6, 4))
plt.bar(["Eğitim Doğruluğu", "Test Doğruluğu"], [train_acc, test_acc], color=["skyblue", "orange"])
plt.title("Model Doğruluğu (KNN)")
plt.ylim(0.8, 1.0)
plt.ylabel("Doğruluk")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 8. ROC Eğrisi ve AUC Skoru
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Eğrisi (KNN)')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Confusion Matrix - Sayısal ve Normalizasyonlu
cm = confusion_matrix(y_test, y_test_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 6))

# Confusion Matrix (Sayısal)
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
plt.title('Confusion Matrix (Sayısal)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

# Confusion Matrix (Normalizasyonlu)
plt.subplot(1, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
plt.title('Confusion Matrix (Normalizasyonlu)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.tight_layout()
plt.show()
