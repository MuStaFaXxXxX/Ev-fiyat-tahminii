from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
from data_preprocessing import preprocess_data
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def load_and_prepare_data():
    try:
        logging.info("Veri seti yükleniyor...")
        data = pd.read_csv('realtor-data.csv')
        
        # Eksik değerleri temizle
        data = data.dropna()
        
        # Kategorik değerleri sayısal değerlere dönüştür
        label_encoders = {}
        categorical_columns = ['state', 'city', 'street', 'status']
        numeric_columns = ['bed', 'bath', 'acre_lot', 'house_size']
        
        for column in categorical_columns:
            if column in data.columns:
                label_encoders[column] = LabelEncoder()
                data[column] = label_encoders[column].fit_transform(data[column])
        
        # Sayısal özellikleri ölçeklendir
        scaler = StandardScaler()
        if all(col in data.columns for col in numeric_columns):
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        
        # Hedef değişkeni ve özellikleri ayır
        feature_columns = categorical_columns + numeric_columns
        X = data[feature_columns]
        y = data['price']
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info(f"Veri seti boyutu: {data.shape}")
        logging.info(f"Eğitim seti boyutu: {X_train.shape}")
        logging.info(f"Test seti boyutu: {X_test.shape}")
        logging.info(f"Kullanılan özellikler: {feature_columns}")
        
        return X_train, X_test, y_train, y_test, label_encoders, scaler
        
    except Exception as e:
        logging.error(f"Veri yükleme hatası: {str(e)}")
        raise

def train_model(X_train, y_train):
    try:
        logging.info("Model eğitimi başlıyor...")
        start_time = time.time()
        
        # Random Forest modelini oluştur ve eğit
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Model eğitimi tamamlandı. Süre: {training_time:.2f} saniye")
        
        return model
        
    except Exception as e:
        logging.error(f"Model eğitimi hatası: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Model değerlendirmesi başlıyor...")
        
        # Tahminleri yap
        y_pred = model.predict(X_test)
        
        # Regresyon metrikleri
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Fiyat aralıklarına göre sınıflandırma
        price_bins = [0, 100000, 200000, 300000, 400000, float('inf')]
        y_test_class = pd.cut(y_test, bins=price_bins, labels=False)
        y_pred_class = pd.cut(y_pred, bins=price_bins, labels=False)
        
        # Sınıflandırma metrikleri
        precision = precision_score(y_test_class, y_pred_class, average='weighted')
        recall = recall_score(y_test_class, y_pred_class, average='weighted')
        f1 = f1_score(y_test_class, y_pred_class, average='weighted')
        
        # Sonuçları logla
        logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logging.info(f"R2 Score: {r2:.2f}")
        logging.info(f"Precision: {precision:.2f}")
        logging.info(f"Recall: {recall:.2f}")
        logging.info(f"F1 Score: {f1:.2f}")
        
        # Görselleştirmeler
        plot_results(y_test, y_pred, y_test_class, y_pred_class, model, X_test)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        logging.error(f"Model değerlendirme hatası: {str(e)}")
        raise

def plot_results(y_test, y_pred, y_test_class, y_pred_class, model, X_test):
    try:
        # Sonuçları görselleştir
        plt.figure(figsize=(15, 10))
        
        # 1. Gerçek vs Tahmin Değerleri
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title('Gerçek vs Tahmin Değerleri')
        
        # 2. Hata Dağılımı
        plt.subplot(2, 2, 2)
        errors = y_test - y_pred
        plt.hist(errors, bins=50)
        plt.xlabel('Hata')
        plt.ylabel('Frekans')
        plt.title('Hata Dağılımı')
        
        # 3. Confusion Matrix
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(y_test_class, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.ylabel('Gerçek Sınıf')
        plt.title('Confusion Matrix')
        
        # 4. Özellik Önemliliği
        plt.subplot(2, 2, 4)
        feature_importance = pd.Series(model.feature_importances_, index=X_test.columns)
        feature_importance.nlargest(10).plot(kind='barh')
        plt.xlabel('Özellik Önemliliği')
        plt.title('En Önemli 10 Özellik')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png')
        plt.close()
        
    except Exception as e:
        logging.error(f"Görselleştirme hatası: {str(e)}")

def save_model_and_encoders(model, label_encoders, scaler):
    try:
        logging.info("Model ve encoder'lar kaydediliyor...")
        
        # Modeli kaydet
        joblib.dump(model, 'price_predictor_model.joblib')
        
        # Encoder'ları kaydet
        for column, encoder in label_encoders.items():
            joblib.dump(encoder, f'{column}_encoder.joblib')
        
        # Scaler'ı kaydet
        joblib.dump(scaler, 'feature_scaler.joblib')
        
        logging.info("Model ve encoder'lar başarıyla kaydedildi")
        
    except Exception as e:
        logging.error(f"Model kaydetme hatası: {str(e)}")
        raise

def main():
    try:
        logging.info("Model eğitim süreci başlatılıyor...")
        
        # Veriyi yükle ve hazırla
        X_train, X_test, y_train, y_test, label_encoders, scaler = load_and_prepare_data()
        
        # Modeli eğit
        model = train_model(X_train, y_train)
        
        # Modeli değerlendir
        metrics = evaluate_model(model, X_test, y_test)
        
        # Modeli ve encoder'ları kaydet
        save_model_and_encoders(model, label_encoders, scaler)
        
        logging.info("Model eğitim süreci başarıyla tamamlandı")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Ana süreç hatası: {str(e)}")
        raise

if __name__ == "__main__":
    main() 