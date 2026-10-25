import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'baselines')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
#RSI > 70 là quá mua, < 30 là quá bán 
#MACD > 0 là xu hướng tăng, < 0 là xu hướng giảm 
#SMA > giá là xu hướng tăng, < giá là xu hướng giảm 
#ATR  càng lớn càng thể hiện sự biến động của thị trường 
#Price Spread (High - Low) thể hiện sự chênh lệch giữa giá cao nhất và giá thấp nhất trong ngày
def prepare_data(df):
    """
    Bước 1 & 2: Xác định Target Variable và Chia tập Train/Test (Chronological)
    """
    print("-> Đang chuẩn bị dữ liệu...")
    
    # TODO 1: Tạo biến mục tiêu (Target) bằng cách shift(-1) cột Log_Return
    # Dùng Log_Return của ngày mai (t+1) làm nhãn để dự đoán cho ngày hôm nay (t)
    #điểm target hôm nay là điểm return vào ngày mai 
    df['Target_Return'] = df['Log_Return'].shift(-1)
    
    # TODO 2: Xử lý vô hạn và drop các dòng bị NaN sinh ra do hàm shift
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # TODO 3: Phân tách X (Features) và y (Target)
    # Loại bỏ các cột không đưa vào thuật toán (Date, Target_Return)
    #tạo danh sách các cột không đưa vào thuật toán
    exclude_cols = ['Date', 'Target_Return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Target_Return']
    
    # TODO 4: Chia dữ liệu Train (80%) / Test (20%) bắt buộc dùng shuffle=False (Chronological)
    # Chia dữ liệu train (80%) và test (20%) theo thời gian, không xáo trộn dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"   + Số mẫu Train: {len(X_train)} | Số mẫu Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def calculate_hit_rate(y_true, y_pred):
    """
    Hàm tính Directional Accuracy (Tỷ lệ đoán trúng hướng tăng/giảm).
    Ý tưởng: Dấu của Thực tế (Tăng +, Giảm -) so sánh với Dấu của Dự đoán.
    """
    # Lấy dấu: 1 (tăng), -1 (giảm), 0 (không đổi)
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    
    # Lọc bỏ những ngày thị trường đứng im (return = 0)
    valid_mask = (true_direction != 0) & (pred_direction != 0)
    
    if np.sum(valid_mask) == 0:
        return 0.0
        
    correct_predictions = (true_direction[valid_mask] == pred_direction[valid_mask]).sum()
    hit_rate = correct_predictions / np.sum(valid_mask)
    
    return hit_rate
#nếu hôm nay tăng thì ngày mai tiếp tục tăng 
def train_naive_model(X_test, y_test):
    """
    Bước 3: Mô hình Naive Forecast (Dự đoán ngày mai = hôm nay)
    """
    print("-> Đang chạy Naive Forecast Baseline...")
    
    # TODO 5: Viết logic dự đoán ngây thơ
    # Do X_test đã chứa cột Log_Return của ngày hôm nay, ta lấy nó làm dự đoán cho ngày mai.
    y_pred_naive = X_test['Log_Return']
    return y_pred_naive

def train_ml_models(X_train, y_train):
    """
    Bước 4: Huấn luyện Linear Regression và XGBoost
    """
    print("-> Đang huấn luyện Linear Regression và XGBoost...")
    
    # TODO 6: Khởi tạo và fit mô hình Linear Regression
    #LR là mô hình hồi quy tuyến tính dựa trên đường thẳng
    # Có thể dùng để tìm mối quan hệ giữa các biến 
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # TODO 7: Khởi tạo và fit mô hình XGBoost
    # Thêm một chút tham số cơ bản chống Overfitting
    # XGBoost là mô hình Boosting dựa trên cây quyết định, có tác dụng trong dự đoán xu hướng giá cổ phiếu 
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return lr_model, xgb_model

def save_models(lr_model, xgb_model, ticker):
    """
    Bước 5: Lưu mô hình bằng joblib
    """
    print("-> Đang lưu mô hình...")
    
    # TODO 8: Dùng joblib.dump để lưu 2 model vào thư mục models/
    joblib.dump(lr_model, os.path.join(MODELS_DIR, f"{ticker}_lr_baseline.pkl"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f"{ticker}_xgb_baseline.pkl"))

def print_and_evaluate(model_name, y_true, y_pred, results_dict):
    # Tính các sai số
    hit_rate = calculate_hit_rate(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    hit_rate_pct = hit_rate * 100
    if hit_rate > 0.5:
        color_text = f"\033[92m{hit_rate_pct:.2f}%\033[0m" # Xanh
    else:
        color_text = f"\033[91m{hit_rate_pct:.2f}%\033[0m" # Đỏ
        
    print(f"   + {model_name:<20}: Hit Rate = {color_text} | RMSE = {rmse:.5f} | MAE = {mae:.5f}")
    results_dict[model_name] = {'Hit Rate': hit_rate_pct, 'RMSE': rmse}

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("BẮT ĐẦU HUẤN LUYỆN BASELINE MODELS CHO TOÀN BỘ VN30\n" + "="*60)
    
    # Tự động quét toàn bộ các mã đã có features trong data/interim/
    tickers = sorted([f.replace('_features.csv', '') for f in os.listdir(INTERIM_DATA_DIR) if f.endswith('_features.csv')])
    print(f"Tổng số mã tìm thấy: {len(tickers)} -> {tickers}")
    
    for ticker in tickers:
        print("\n" + "#"*60)
        print(f" ĐANG XỬ LÝ MÃ: {ticker}")
        print("#"*60)
        
        data_path = os.path.join(INTERIM_DATA_DIR, f"{ticker}_features.csv")
        if not os.path.exists(data_path):
            print(f"Không tìm thấy file {data_path}. Bỏ qua mã này.")
            continue
            
        df = pd.read_csv(data_path)
        print(f"-> Đã nạp file {ticker} thành công! Kích thước ban đầu: {df.shape}")
        
        # 1. Chuẩn bị dữ liệu
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # 2. Chạy Naive Baseline
        y_pred_naive = train_naive_model(X_test, y_test)
        
        # 3. Huấn luyện ML
        lr_model, xgb_model = train_ml_models(X_train, y_train)
        
        # 4. Dự đoán
        y_pred_lr = lr_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        
        # 5. Lưu mô hình
        save_models(lr_model, xgb_model, ticker)
        
        # 6. Đánh giá
        results = {}
        print("\nKẾT QUẢ ĐÁNH GIÁ (HIT RATE, RMSE, MAE) TRÊN TẬP TEST:")
        print_and_evaluate("Naive Forecast", y_test, y_pred_naive, results)
        print_and_evaluate("Linear Regression", y_test, y_pred_lr, results)
        print_and_evaluate("XGBoost", y_test, y_pred_xgb, results)
        
        # 7. Vẽ biểu đồ
        print("\n-> Đang vẽ biểu đồ so sánh và lưu vào reports/figures/...")
        models = list(results.keys())
        hit_rates = [results[m]['Hit Rate'] for m in models]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(models, hit_rates, color=['#e74c3c', '#3498db', '#2ecc71'])
        plt.axhline(y=50, color='gray', linestyle='--', label='50% (Random Guessing)')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center', fontweight='bold')
            
        plt.title(f"So sánh Hit Rate các thuật toán ({ticker})")
        plt.ylabel("Hit Rate (%)")
        plt.ylim(40, 60)
        plt.legend()
        
        plot_path = os.path.join(REPORTS_DIR, f"{ticker}_hit_rate_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   + Đã lưu biểu đồ: {ticker}_hit_rate_comparison.png")
        
    print("\n" + "="*60)
    print("HOÀN THÀNH HUẤN LUYỆN TOÀN BỘ CÁC MÃ CỔ PHIẾU!")
