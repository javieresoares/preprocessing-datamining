import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Memuat dataset
train_df = pd.read_csv('dataset/train.csv', sep=';')
test_df = pd.read_csv('dataset/test.csv', sep=';')

# Menampilkan beberapa baris pertama dari dataset
print("Dataset Latih:")
print(train_df.head())
print("\nDataset Uji:")
print(test_df.head())

# Memahami data
# Memeriksa bentuk dataset
print("\nBentuk Dataset Latih:", train_df.shape)
print("Bentuk Dataset Uji:", test_df.shape)

# Memeriksa nilai yang hilang
print("\nNilai yang Hilang di Dataset Latih:\n", train_df.isnull().sum())
print("\nNilai yang Hilang di Dataset Uji:\n", test_df.isnull().sum())

# Mengencode variabel target
train_df['y'] = train_df['y'].map({'yes': 1, 'no': 0})
test_df['y'] = test_df['y'].map({'yes': 1, 'no': 0})

# Memisahkan fitur dan variabel target
X_train = train_df.drop(columns=['y'])
y_train = train_df['y']
X_test = test_df.drop(columns=['y'])
y_test = test_df['y']

# Mengidentifikasi fitur kategorikal dan numerik
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Langkah-langkah preprocessing
# Mendefinisikan preprocessing untuk fitur numerik dan kategorikal
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Melakukan scaling pada fitur numerik
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding untuk fitur kategorikal
])

# Menggabungkan langkah-langkah preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Melakukan preprocessing pada data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Mengonversi hasil preprocessing ke DataFrame untuk memudahkan penanganan selanjutnya (opsional)
X_train_processed = pd.DataFrame(X_train_processed)
X_test_processed = pd.DataFrame(X_test_processed)

# Menampilkan bentuk data yang telah diproses
print("\nBentuk Fitur Latih yang Diproses:", X_train_processed.shape)
print("Bentuk Fitur Uji yang Diproses:", X_test_processed.shape)

# Menyimpan data yang telah diproses ke CSV (opsional)
X_train_processed.to_csv('dataset/X_train_processed.csv', index=False)
y_train.to_csv('dataset/y_train_processed.csv', index=False)
X_test_processed.to_csv('dataset/X_test_processed.csv', index=False)
y_test.to_csv('dataset/y_test_processed.csv', index=False)

print("\nProses preprocessing selesai dengan sukses.")
