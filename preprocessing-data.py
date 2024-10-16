import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Membaca dataset dengan delimiter titik koma
dataset = pd.read_csv('Data.csv', sep=';')

# Memisahkan fitur (X) dan target (y)
X = dataset.iloc[:, 1:-1].values  # Mengambil semua kolom kecuali kolom pertama dan terakhir
y = dataset.iloc[:, -1].values  # Mengambil kolom terakhir sebagai target

print("X_train before preprocessing:", X)
print("y_train before preprocessing:", y)

# Mengatasi missing values menggunakan SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:3])  # Menangani kolom 'Salary' (kolom 3)
X[:, 2:3] = imputer.transform(X[:, 2:3])  # Mengimputasi nilai yang hilang

print("X after imputing missing values:", X)

# Encoding fitur kategori
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # Kolom 'Country' (kolom 1)
X = np.array(ct.fit_transform(X))  # Mengubah data menjadi array NumPy

print("X after encoding categorical features:", X)

# Encoding variabel target (y)
le = LabelEncoder()
y = le.fit_transform(y)  # Mengubah variabel target menjadi format numerik

print("y after encoding:", y)

# Memisahkan dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

# Feature Scaling (Standardisasi)
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])  # Melakukan scaling untuk kolom numerik mulai dari kolom ke-4
X_test[:, 3:] = sc.transform(X_test[:, 3:])  # Menggunakan scaling yang sama untuk dataset uji

print("X_train after scaling:", X_train)
print("X_test after scaling:", X_test)
