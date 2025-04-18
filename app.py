import sys
import pandas as pd
import joblib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QLineEdit, QPushButton,
                            QTextEdit, QMessageBox, QListWidget, QComboBox,
                            QTabWidget, QCompleter, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class SearchLineEdit(QLineEdit):
    def __init__(self, parent=None, completer_items=None):
        super().__init__(parent)
        if completer_items:
            completer = QCompleter(completer_items)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.setCompleter(completer)
        
        self.textChanged.connect(self.on_text_changed)
    
    def on_text_changed(self, text):
        if hasattr(self.parent(), 'update_suggestions'):
            self.parent().update_suggestions(self.objectName(), text)

class PricePredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emlak Fiyat Tahmin Uygulaması")
        self.setGeometry(100, 100, 1200, 800)
        
        try:
            # Veri setini yükle ve hazırla
            self.data = pd.read_csv('realtor-data.csv')
            self.prepare_data()
            
            # Model ve encoder'ları yükle
            self.model = joblib.load('price_predictor_model.joblib')
            self.state_encoder = joblib.load('state_encoder.joblib')
            self.city_encoder = joblib.load('city_encoder.joblib')
            self.street_encoder = joblib.load('street_encoder.joblib')
            self.status_encoder = joblib.load('status_encoder.joblib')
            self.bed_encoder = joblib.load('bed_encoder.joblib')
            self.bath_encoder = joblib.load('bath_encoder.joblib')
            self.acre_lot_encoder = joblib.load('acre_lot_encoder.joblib')
            self.house_size_encoder = joblib.load('house_size_encoder.joblib')
            self.scaler = joblib.load('feature_scaler.joblib')
            
            # Benzersiz değerleri al
            self.unique_states = sorted(self.data['state'].unique())
            self.unique_cities = sorted(self.data['city'].unique())
            self.unique_streets = sorted(self.data['street'].unique())
            self.unique_status = sorted(self.data['status'].unique())
            
            print(f"Benzersiz eyalet sayısı: {len(self.unique_states)}")
            print(f"Benzersiz şehir sayısı: {len(self.unique_cities)}")
            print(f"Benzersiz sokak sayısı: {len(self.unique_streets)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri yüklenirken hata oluştu: {str(e)}")
            sys.exit(1)
        
        self.init_ui()
    
    def prepare_data(self):
        try:
            # Eksik değerleri temizle
            self.data = self.data.dropna(subset=['state', 'city', 'street', 'price', 'status', 'bed', 'bath', 'acre_lot', 'house_size'])
            
            # Tüm değerleri string'e dönüştür ve temizle
            self.data['state'] = self.data['state'].astype(str).str.strip()
            self.data['city'] = self.data['city'].astype(str).str.strip()
            self.data['street'] = self.data['street'].astype(str).str.strip()
            self.data['status'] = self.data['status'].astype(str).str.strip()
            
            # Fiyat sütununu sayısal değere dönüştür
            self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')
            
            # NaN değerleri temizle
            self.data = self.data.dropna()
            
        except Exception as e:
            print(f"Veri hazırlama sırasında hata: {str(e)}")
            raise
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Ana başlık
        title_label = QLabel("Emlak Fiyat Tahmin Uygulaması")
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Giriş alanları
        input_layout = QVBoxLayout()
        
        # Eyalet girişi
        state_layout = QHBoxLayout()
        state_layout.addWidget(QLabel("Eyalet:"))
        self.state_input = SearchLineEdit(completer_items=self.unique_states)
        self.state_input.setObjectName('state')
        self.state_input.setPlaceholderText("Eyalet adı girin veya seçin...")
        state_layout.addWidget(self.state_input)
        input_layout.addLayout(state_layout)
        
        # Şehir girişi
        city_layout = QHBoxLayout()
        city_layout.addWidget(QLabel("Şehir:"))
        self.city_input = SearchLineEdit(completer_items=self.unique_cities)
        self.city_input.setObjectName('city')
        self.city_input.setPlaceholderText("Şehir adı girin veya seçin...")
        city_layout.addWidget(self.city_input)
        input_layout.addLayout(city_layout)
        
        # Sokak girişi
        street_layout = QHBoxLayout()
        street_layout.addWidget(QLabel("Sokak:"))
        self.street_input = SearchLineEdit(completer_items=self.unique_streets)
        self.street_input.setObjectName('street')
        self.street_input.setPlaceholderText("Sokak adı girin veya seçin...")
        street_layout.addWidget(self.street_input)
        input_layout.addLayout(street_layout)
        
        # Durum girişi
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Durum:"))
        self.status_input = QComboBox()
        self.status_input.addItems(self.unique_status)
        status_layout.addWidget(self.status_input)
        input_layout.addLayout(status_layout)
        
        # Yatak sayısı
        bed_layout = QHBoxLayout()
        bed_layout.addWidget(QLabel("Yatak Sayısı:"))
        self.bed_input = QSpinBox()
        self.bed_input.setRange(1, 10)
        bed_layout.addWidget(self.bed_input)
        input_layout.addLayout(bed_layout)
        
        # Banyo sayısı
        bath_layout = QHBoxLayout()
        bath_layout.addWidget(QLabel("Banyo Sayısı:"))
        self.bath_input = QDoubleSpinBox()
        self.bath_input.setRange(1, 10)
        self.bath_input.setSingleStep(0.5)
        bath_layout.addWidget(self.bath_input)
        input_layout.addLayout(bath_layout)
        
        # Arazi büyüklüğü
        acre_layout = QHBoxLayout()
        acre_layout.addWidget(QLabel("Arazi Büyüklüğü (Acre):"))
        self.acre_input = QDoubleSpinBox()
        self.acre_input.setRange(0.1, 100)
        self.acre_input.setSingleStep(0.1)
        acre_layout.addWidget(self.acre_input)
        input_layout.addLayout(acre_layout)
        
        # Ev büyüklüğü
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Ev Büyüklüğü (sqft):"))
        self.size_input = QSpinBox()
        self.size_input.setRange(100, 10000)
        self.size_input.setSingleStep(100)
        size_layout.addWidget(self.size_input)
        input_layout.addLayout(size_layout)
        
        layout.addLayout(input_layout)
        
        # Tahmin butonu
        predict_btn = QPushButton("Fiyat Tahmini Yap")
        predict_btn.clicked.connect(self.predict_price)
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(predict_btn)
        
        # Sonuç alanı
        self.result_label = QLabel("Tahmin edilen fiyat: ")
        self.result_label.setFont(QFont('Arial', 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        # Veri gösterim alanı
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setFont(QFont('Arial', 12))
        self.data_display.setMinimumHeight(300)
        layout.addWidget(self.data_display)
        
        # Stil ayarları
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 13px;
                color: #333;
                margin: 5px;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 13px;
                min-width: 300px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 13px;
                min-width: 300px;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 13px;
                min-width: 100px;
            }
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
            }
        """)
    
    def update_suggestions(self, field_name, text):
        if not text:
            return
        
        try:
            # Seçilen alana göre filtreleme yap
            if field_name == 'state':
                filtered_data = self.data[self.data['state'].str.lower().str.startswith(text.lower())]
                if not filtered_data.empty:
                    unique_cities = filtered_data['city'].unique()
                    self.city_input.completer().setModel(QCompleter(sorted(unique_cities)))
            
            elif field_name == 'city':
                filtered_data = self.data[self.data['city'].str.lower().str.startswith(text.lower())]
                if not filtered_data.empty:
                    unique_streets = filtered_data['street'].unique()
                    self.street_input.completer().setModel(QCompleter(sorted(unique_streets)))
            
            # Seçilen değere göre verileri göster
            self.show_related_data(field_name, text)
            
        except Exception as e:
            print(f"Öneri güncellenirken hata: {str(e)}")
    
    def show_related_data(self, field_name, value):
        try:
            # Seçilen değere göre verileri filtrele
            filtered_data = self.data[self.data[field_name].str.lower().str.startswith(value.lower())]
            
            if filtered_data.empty:
                self.data_display.setText(f"{field_name} için veri bulunamadı.")
                return
            
            # İstatistikleri hesapla
            avg_price = filtered_data['price'].mean()
            min_price = filtered_data['price'].min()
            max_price = filtered_data['price'].max()
            count = len(filtered_data)
            
            display_text = f"{field_name.capitalize()} '{value}' için istatistikler:\n\n"
            display_text += f"Toplam Kayıt: {count:,}\n"
            display_text += f"Ortalama Fiyat: ${avg_price:,.2f}\n"
            display_text += f"Minimum Fiyat: ${min_price:,.2f}\n"
            display_text += f"Maksimum Fiyat: ${max_price:,.2f}\n"
            display_text += "\nÖrnek Kayıtlar:\n" + "-"*50 + "\n"
            
            # İlk 5 kaydı göster
            for _, row in filtered_data.head().iterrows():
                display_text += f"Eyalet: {row['state']}\n"
                display_text += f"Şehir: {row['city']}\n"
                display_text += f"Sokak: {row['street']}\n"
                display_text += f"Durum: {row['status']}\n"
                display_text += f"Yatak: {row['bed']}\n"
                display_text += f"Banyo: {row['bath']}\n"
                display_text += f"Arazi: {row['acre_lot']} acre\n"
                display_text += f"Büyüklük: {row['house_size']} sqft\n"
                display_text += f"Fiyat: ${row['price']:,.2f}\n"
                display_text += "-"*50 + "\n"
            
            self.data_display.setText(display_text)
            
        except Exception as e:
            print(f"Veri gösterimi sırasında hata: {str(e)}")
            self.data_display.setText(f"Hata oluştu: {str(e)}")
    
    def predict_price(self):
        try:
            # Girdileri al
            state = self.state_input.text()
            city = self.city_input.text()
            street = self.street_input.text()
            status = self.status_input.currentText()
            bed = self.bed_input.value()
            bath = self.bath_input.value()
            acre_lot = self.acre_input.value()
            house_size = self.size_input.value()
            
            if not all([state, city, street]):
                QMessageBox.warning(self, "Uyarı", "Lütfen en azından eyalet, şehir ve sokak bilgilerini girin!")
                return
            
            # Kategorik değerleri dönüştür
            state_encoded = self.state_encoder.transform([state])[0]
            city_encoded = self.city_encoder.transform([city])[0]
            street_encoded = self.street_encoder.transform([street])[0]
            status_encoded = self.status_encoder.transform([status])[0]
            
            # Sayısal değerleri ölçeklendir
            numeric_features = np.array([[bed, bath, acre_lot, house_size]])
            scaled_numeric = self.scaler.transform(numeric_features)
            
            # Tüm özellikleri birleştir
            features = np.array([[
                state_encoded,
                city_encoded,
                street_encoded,
                status_encoded,
                scaled_numeric[0][0],  # bed
                scaled_numeric[0][1],  # bath
                scaled_numeric[0][2],  # acre_lot
                scaled_numeric[0][3]   # house_size
            ]])
            
            # Tahmin yap
            prediction = self.model.predict(features)[0]
            
            # Seçilen konumdaki gerçek fiyatları bul
            actual_prices = self.data[
                (self.data['state'] == state) &
                (self.data['city'] == city) &
                (self.data['street'] == street)
            ]['price']
            
            result_text = f"Tahmin edilen fiyat: ${prediction:,.2f}\n\n"
            
            if not actual_prices.empty:
                avg_price = actual_prices.mean()
                result_text += f"Bu konumdaki ortalama fiyat: ${avg_price:,.2f}\n"
                result_text += f"Fiyat farkı: ${abs(prediction - avg_price):,.2f}"
            
            self.result_label.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Tahmin yapılırken hata oluştu: {str(e)}")
            print(f"Hata detayı: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PricePredictorApp()
    window.show()
    sys.exit(app.exec_()) 