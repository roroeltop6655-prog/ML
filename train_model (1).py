import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. تحميل البيانات
df = pd.read_csv('egypt_train_data.csv')

# 2. تحويل البيانات النصية لأرقام (Encoding)
le_source = LabelEncoder()
le_dest = LabelEncoder()
le_type = LabelEncoder()

df['Source_Encoded'] = le_source.fit_transform(df['Source'])
df['Destination_Encoded'] = le_dest.fit_transform(df['Destination'])
df['Train_Type_Encoded'] = le_type.fit_transform(df['Train_Type'])

# 3. تحديد المدخلات (X) والمخرجات (y)
X = df[['Source_Encoded', 'Destination_Encoded', 'Distance_KM', 'Train_Type_Encoded']]
y = df['Travel_Time_Hours']

# 4. تقسيم البيانات (للتدريب والاختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. بناء الموديل (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. حفظ الموديل والمحولات (Encoders) لاستخدامها لاحقاً
joblib.dump(model, 'train_arrival_model.pkl')
joblib.dump(le_source, 'le_source.pkl')
joblib.dump(le_dest, 'le_dest.pkl')
joblib.dump(le_type, 'le_type.pkl')

print("Model trained and saved successfully!")
