import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set_style("whitegrid")

def main():
    print("=== GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU ===")

    df = pd.read_csv('data/Wiki_Page_views.csv')
    
    row_index = 3
    page_name = df.iloc[row_index, 0]
    print(f"Đang xử lý bài viết: {page_name}")
    
    df_target = df.iloc[row_index, 1:].to_frame(name='Views')
    df_target.index = pd.to_datetime(df_target.index, format='%Y%m%d%H', errors='coerce')
    df_target['Views'] = pd.to_numeric(df_target['Views'], errors='coerce')

    df_target['Views'] = df_target['Views'].ffill()

    print("\n=== GIAI ĐOẠN 2: THỐNG KÊ MÔ TẢ ===")
    print(df_target.describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(df_target['Views'], bins=30, kde=True, color='purple')
    plt.title('Daily Page Views Distribution (Histogram)', fontsize=14, fontweight='bold')
    plt.xlabel('Page Views')
    plt.ylabel('Frequency')
    plt.savefig('images/histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== GIAI ĐOẠN 3: KIỂM ĐỊNH TÍNH DỪNG ===")

    decomposition = seasonal_decompose(df_target['Views'], model='additive', period=7)
    fig = decomposition.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle('Trend and Seasonality Decomposition of Wikipedia Traffic', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('images/decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    result = adfuller(df_target['Views'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    print("\n=== GIAI ĐOẠN 4: CHỌN THAM SỐ & ARMA ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(df_target['Views'], lags=40, ax=axes[0], color='blue')
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Correlation')
    
    plot_pacf(df_target['Views'], lags=40, ax=axes[1], color='red')
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')
    plt.tight_layout()
    plt.savefig('images/acf_pacf.png', dpi=300, bbox_inches='tight')
    plt.close()

    train = df_target.iloc[:-30]
    test = df_target.iloc[-30:]

    print("Huấn luyện mô hình ARMA(2,2)...")
    model_arma = ARIMA(train['Views'], order=(2, 0, 2))
    results_arma = model_arma.fit()
    predictions_arma = results_arma.forecast(steps=30)
    
    plt.figure(figsize=(14, 6))
    plt.plot(train.index[-60:], train.iloc[-60:]['Views'], label='Training Data', color='green')
    plt.plot(test.index, test['Views'], label='Actual Data (Test)', color='blue')
    plt.plot(test.index, predictions_arma, label='ARMA(2,2) Forecast', color='red', linestyle='--', marker='o', markersize=4)
    plt.title('Wikipedia Traffic Forecasting (Last 30 Days) using ARMA(2,2)', fontsize=15, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Page Views')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/arma_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== GIAI ĐOẠN 5: TỐI ƯU HÓA BẰNG SARIMA ===")
    print("Huấn luyện mô hình SARIMA(1,0,1)(1,0,1,7)...")
    model_sarima = SARIMAX(train['Views'], order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    results_sarima = model_sarima.fit(disp=False)
    print(results_sarima.summary())
    predictions_sarima = results_sarima.forecast(steps=30)

    plt.figure(figsize=(14, 6))
    plt.plot(train.index[-60:], train.iloc[-60:]['Views'], label='Training Data', color='green')
    plt.plot(test.index, test['Views'], label='Actual Data (Test)', color='blue')
    plt.plot(test.index, predictions_sarima, label='SARIMA Forecast (7-day cycle)', color='red', linestyle='--', marker='o', markersize=4)
    plt.title('Wikipedia Traffic Forecasting using SARIMA', fontsize=15, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Page Views')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/sarima_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nQuá trình hoàn tất! Tất cả biểu đồ đã được lưu vào thư mục 'images/'.")

if __name__ == "__main__":
    main()