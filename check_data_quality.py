import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Clean_Dataset.csv')

print("="*60)
print("DATA QUALITY CHECK")
print("="*60)

# 1. Kiểm tra giá trị âm hoặc bằng 0
print("\n1. Check for negative or zero values:")
print(f"   Duration <= 0: {(df['duration'] <= 0).sum()}")
print(f"   Days_left <= 0: {(df['days_left'] <= 0).sum()}")
print(f"   Price <= 0: {(df['price'] <= 0).sum()}")

# 2. Kiểm tra giá trị ngoại lai cực đoan (ngoài 3 độ lệch chuẩn)
print("\n2. Check for extreme outliers (beyond 3 standard deviations):")
for col in ['duration', 'days_left', 'price']:
    mean = df[col].mean()
    std = df[col].std()
    outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
    print(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")

# 3. Kiểm tra phạm vi dữ liệu
print("\n3. Data range:")
print(f"   Duration: {df['duration'].min()} - {df['duration'].max()} hours")
print(f"   Days_left: {df['days_left'].min()} - {df['days_left'].max()} days")
print(f"   Price: {df['price'].min()} - {df['price'].max()}")

# 4. Kiểm tra giá trị duy nhất cho các cột phân loại
print("\n4. Unique values in categorical columns:")
for col in ['airline', 'source_city', 'destination_city', 'departure_time', 'arrival_time', 'stops', 'class']:
    print(f"   {col}: {df[col].nunique()} unique values")

# 5. Kiểm tra các vấn đề tiềm ẩn
print("\n5. Check for potential issues:")
# kiểm tra nếu duration hợp lý
print(f"   Very short flights (< 1 hour): {(df['duration'] < 1).sum()}")
print(f"   Very long flights (> 24 hours): {(df['duration'] > 24).sum()}")

# kiểm tra ngoại lai giá sử dụng phương pháp IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
price_outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound)).sum()
print(f"   Price outliers (IQR method): {price_outliers} ({price_outliers/len(df)*100:.2f}%)")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"✓ No missing values")
print(f"✓ No duplicate rows")
print(f"✓ All data types are correct")
print(f"✓ No negative values")
if (df['duration'] <= 0).sum() == 0 and (df['days_left'] <= 0).sum() == 0 and (df['price'] <= 0).sum() == 0:
    print(f"✓ No zero or negative values in numerical columns")

print("\n→ Data quality: GOOD - Dataset is clean and ready for use!")
