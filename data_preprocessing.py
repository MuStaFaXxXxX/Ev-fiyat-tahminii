import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import time

def preprocess_data():
    try:
        start_time = time.time()
        print("Veri dosyası okunuyor...")
        
        # Veriyi daha verimli okuma
        df = pd.read_csv('realtor-data.csv', usecols=['price', 'state', 'city', 'street'])
        print(f"Veri başarıyla okundu. Toplam satır sayısı: {len(df)}")
        
        # Temel veri temizleme
        print("Eksik değerler temizleniyor...")
        df = df.dropna()
        print(f"Temizleme sonrası satır sayısı: {len(df)}")
        
        # Kategorik değişkenleri sayısallaştır
        print("Kategorik değişkenler sayısallaştırılıyor...")
        label_encoders = {}
        categorical_columns = ['state', 'city', 'street']
        
        for column in categorical_columns:
            print(f"{column} sütunu işleniyor...")
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
        
        # Eğitim ve test verilerini ayır
        print("Veri seti bölünüyor...")
        X = df[['state', 'city', 'street']].astype(np.int32)  # Bellek optimizasyonu
        y = df['price'].astype(np.float32)  # Bellek optimizasyonu
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Eğitim seti boyutu: {len(X_train)}, Test seti boyutu: {len(X_test)}")
        
        # Label encoder'ları kaydet
        print("Encoder'lar kaydediliyor...")
        for column, encoder in label_encoders.items():
            joblib.dump(encoder, f'{column}_encoder.joblib')
            print(f"{column}_encoder.joblib kaydedildi")
        
        processing_time = time.time() - start_time
        print(f"Veri ön işleme tamamlandı. Toplam süre: {processing_time:.2f} saniye")
        
        return X_train, X_test, y_train, y_test, label_encoders
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data()
        print("Veri ön işleme başarıyla tamamlandı!")
    except Exception as e:
        print(f"Veri ön işleme sırasında hata oluştu: {str(e)}") 