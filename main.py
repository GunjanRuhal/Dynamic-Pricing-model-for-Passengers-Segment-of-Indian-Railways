import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from scipy.stats.mstats import winsorize
from xgboost import XGBRegressor, plot_importance
import shap
%matplotlib inline

# DATA LOADING & PRE-PROCESSING
df_new=pd.read_csv("last_4months_2024.csv")
# df_new.head()
# df_new['booking_date']
# df_new.isnull().sum()

# Load and process holiday data
holiday = "Holiday table 2023_2025.xlsx"
df_holiday = pd.read_excel(holiday)
df_holiday.to_csv("holiday.csv", index=False)
print("Conversion complete: 'holiday_sheet.csv' created.")

# df_holiday.head()
# df_holiday.describe()
# df_holiday.isnull().sum()
list1=df_holiday['Holiday_Flag'].tolist()
count=0
for ls in list1:
    count+=1
print(count)

# Station to State Mapping
station_to_state = {
    'KOTA': 'Rajasthan', 'RTM': 'Madhya Pradesh', 'MTJ': 'Uttar Pradesh',
    'AGC': 'Uttar Pradesh', 'GWL': 'Madhya Pradesh', 'VGLJ': 'Gujarat',
    'BRC': 'Gujarat', 'ST': 'Gujarat', 'BVI': 'Maharashtra', 'MMCT': 'Maharashtra',
    'BL': 'Gujarat', 'VAPI': 'Gujarat', 'BH': 'Rajasthan', 'JL': 'Rajasthan',
    'NK': 'Maharashtra', 'KYN': 'Maharashtra', 'CSMT': 'Maharashtra', 'NAD': 'Maharashtra',
    'SWM': 'Rajasthan', 'BPL': 'Madhya Pradesh', 'BSL': 'Maharashtra',
    'MAS': 'Tamil Nadu', 'BZA': 'Andhra Pradesh', 'NGP': 'Maharashtra',
    'BPQ': 'Maharashtra', 'WL': 'Andhra Pradesh', 'RKMP': 'Madhya Pradesh',
    'NZM': 'Delhi', 'NDLS': 'Delhi'
}

# Clean state column
df_holiday['State'] = df_holiday['State'].astype(str).str.strip().str.title()

# Convert dates to datetime.date for merging
df_new['journey_date'] = pd.to_datetime(df_new['journey_date']).dt.date
df_holiday['Journey_Date'] = pd.to_datetime(df_holiday['Journey_Date']).dt.date

# Map boarding station to state
df_new['boarding_state'] = df_new['brdpt_code'].map(station_to_state)
df_new['boarding_state'] = df_new['boarding_state'].astype(str).str.strip().str.title()

# Check for unmapped stations
unmapped = df_new[df_new['boarding_state'].isnull()]['brdpt_code'].unique()
if len(unmapped) > 0:
    print("Unmapped station codes:", unmapped)

# Merge datasets on date and state
merged_df = pd.merge(
    df_new,
    df_holiday[['Journey_Date', 'State', 'Holiday_Flag', 'Holiday_Name']],
    how='left',
    left_on=['journey_date', 'boarding_state'],
    right_on=['Journey_Date', 'State']
)

print(df_holiday.columns.tolist())  # Check holiday columns

# Create binary holiday flag
merged_df['is_holiday'] = merged_df['Holiday_Flag'].notnull().astype(int)

# merged_df.head()
# merged_df.info()
# merged_df.describe()
# merged_df.isnull().sum()

# Rename and initialize fare/type columns
merged_df['base_fare'] = merged_df['cashbasefare']
merged_df['total_fare'] = merged_df['fi_earning']
merged_df['service_charge'] = 0
merged_df['reservation_fee'] = 0
merged_df['train_type'] = merged_df['rep_trn_type_nw'].astype(str)
merged_df['coach_type'] = merged_df['cls']

# FEATURE ENGINEERING (Temporal & Lead Time)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'

# Ensure journey_date is datetime
merged_df['journey_date'] = pd.to_datetime(merged_df['journey_date'])

# Apply season function
merged_df['season'] = merged_df['journey_date'].dt.month.apply(get_season)

print(f"Original rows: {len(df_new)}")
print(f"Merged rows:    {len(merged_df)}")
# merged_df.head()

merged_df['booking_date'] = pd.to_datetime(merged_df['booking_date'])
merged_df['journey_date'] = pd.to_datetime(merged_df['journey_date'])
merged_df['booking_lead_time'] = (merged_df['journey_date'] - merged_df['booking_date']).dt.days
merged_df['is_weekend'] = merged_df['journey_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

print(merged_df.columns.tolist())

# Drop redundant/engineered columns
merged_df.drop(columns=[
    'cashbasefare',
    'fi_earning',
    'cls',
    'rep_trn_type_nw',
    'Journey_Date',
    'State',
    'Holiday_Name','Journey_Date_y', 'State_y', 'Holiday_Name_y', 'Journey_Date_x', 'State_x','Holiday_Name_x', 'train_type'
], inplace=True, errors='ignore')

print(merged_df.columns.tolist())

# merged_df.info()
# merged_df.isnull().sum()
list1=merged_df['is_holiday'].tolist()

count=0
for ls in list1:
    if(ls==1):
        count+=1
print(count)

# Final cleaning/preparation
merged_df.dropna(subset=['trnno'], inplace=True)
merged_df.drop_duplicates(inplace=True)
print(merged_df.isnull().sum())

merged_df['Holiday_Flag'] = merged_df['Holiday_Flag'].fillna(0).astype(int)
print(merged_df.isnull().sum())


# FEATURE ENGINEERING (Encoding)
from sklearn.preprocessing import LabelEncoder
# Copy to avoid modifying original dataframe
encoded_df = merged_df.copy()

cat_cols = [
    'brdpt_code', 'resupto_code', 'trn_type',
    'boarding_state',
    'trn_type', 'coach_type', 'season'
]

# Apply Label Encoding
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
    label_encoders[col] = le

print(encoded_df[cat_cols].head())

encoded_df.to_csv("final_encoded_dataset.csv", index=False)

# encoded_df.describe()
# encoded_df.isnull().sum()


# EXPLORATORY DATA ANALYSIS (EDA)
encoded_df = pd.read_csv("final_encoded_dataset.csv")

# CORRELATIONAL ANALYSIS
numeric_df = encoded_df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("Total fare and base fare are highly correlated.")

# BIVARIATE ANALYSIS
sns.histplot(encoded_df['total_fare'], kde=True)
plt.title("Distribution of Total Fare")
plt.show()

sns.scatterplot(data=encoded_df, x='booking_lead_time', y='total_fare')
plt.title("Booking Lead Time vs Total Fare")
plt.show()

sns.lmplot(data=encoded_df, x='booking_lead_time', y='total_fare', line_kws={"color": "red"})
plt.title("Booking Lead Time vs Total Fare with Regression Line")
plt.show()
#If the points go upward → Fares increase with longer lead time.

plt.figure(figsize=(10, 6))
sns.histplot(data=encoded_df, x='booking_lead_time', y='total_fare', bins=50, cbar=True)
plt.title("Booking Lead Time vs Total Fare (2D Histogram)")
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=encoded_df,
    x='booking_lead_time',
    y='total_fare',
    fill=True,
    cmap='magma',
    thresh=0.05
)
plt.title("2D KDE: Booking Lead Time vs Total Fare")
plt.xlabel('Booking Lead Time')
plt.ylabel('Total Fare')
plt.show()

sns.boxplot(data=encoded_df, x='coach_type', y='total_fare')
plt.title("Fare by Coach Type")
plt.show()

# Invert LabelEncoder to get original labels
original_labels = label_encoders['coach_type'].inverse_transform(encoded_df['coach_type'].unique())
print(dict(zip(encoded_df['coach_type'].unique(), original_labels)))

# FEATURE ENGINEERING (Interactions & Demand)
encoded_df['coach_type_x_train_type'] = encoded_df['coach_type'] * encoded_df['trn_type']
encoded_df['lead_time_x_is_holiday'] = encoded_df['booking_lead_time'] * encoded_df['is_holiday']
encoded_df['is_holiday_and_weekend'] = encoded_df['is_holiday'] & encoded_df['is_weekend']

encoded_df['journey_month'] = pd.to_datetime(encoded_df['journey_date']).dt.month
encoded_df['journey_week'] = pd.to_datetime(encoded_df['journey_date']).dt.isocalendar().week.astype(int)

train_counts = encoded_df['trnno'].value_counts().to_dict()
encoded_df['train_demand_score'] = encoded_df['trnno'].map(train_counts)

# ONE HOT ENCODING
desired_categorical_columns = [
    'train_type', 'brdpt_code', 'resupto_code',
    'boarding_state', 'destination_state', 'coach_type', 'season'
]
available_categorical_columns = [col for col in desired_categorical_columns if col in encoded_df.columns]
encoded_df = pd.get_dummies(encoded_df, columns=available_categorical_columns, drop_first=True, dtype=int)

# Save to CSV
encoded_df.to_csv("final_encoded_dataset_with_onehot.csv", index=False)

train_df=pd.read_csv('final_encoded_dataset_with_onehot.csv')
train_popularity = train_df['trnno'].value_counts(normalize=True)
train_df['train_popularity'] = train_df['trnno'].map(train_popularity)

print(train_df.describe())

num_cols = [
    'total_fare', 'base_fare', 'booking_lead_time',
    'train_demand_score'
]

for col in num_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train_df, x=col)
    plt.title(f'Outlier Detection for {col}')
    plt.show()

# OUTLIER DETECTION
# 3. Outlier count using IQR method
def outlier_count(col):
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower) | (train_df[col] > upper)]
    print(f'{col}: {len(outliers)} outliers')

for col in num_cols:
    outlier_count(col)

# Filter negative values and make a copy to avoid SettingWithCopyWarning
train_df = train_df[
    (train_df['total_fare'] >= 0) &
    (train_df['base_fare'] >= 0) &
    (train_df['booking_lead_time'] >= 0)
].copy()

# Apply log1p safely to avoid divide-by-zero warning
train_df['total_fare_log'] = np.log1p(train_df['total_fare'])
train_df['base_fare_log'] = np.log1p(train_df['base_fare'])
train_df['booking_lead_time_log'] = np.log1p(train_df['booking_lead_time'])

# Visualize boxplots for original and log-transformed features
cols_to_check = ['total_fare', 'base_fare', 'booking_lead_time', 'train_demand_score']
log_cols = ['total_fare_log', 'base_fare_log', 'booking_lead_time_log']

for col in cols_to_check:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train_df, x=col)
    plt.title(f'Outlier Detection for {col}')
    plt.show()

for col in log_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train_df, x=col)
    plt.title(f'Outlier Detection (Log-Transformed) for {col}')
    plt.show()

# encoded_df.columns

from scipy.stats.mstats import winsorize
import seaborn as sns
import matplotlib.pyplot as plt

# Apply asymmetric Winsorization based on visual outliers
# 3% left-tail (many 0s), 1% right-tail
train_df['total_fare_win'] = winsorize(train_df['total_fare'], limits=[0.03, 0.01])
train_df['base_fare_win'] = winsorize(train_df['base_fare'], limits=[0.03, 0.01])

# Plot for total_fare_win
plt.figure(figsize=(8, 4))
sns.boxplot(data=train_df, x='total_fare_win', color='skyblue')
plt.title('Boxplot: Winsorized Total Fare')
plt.xlabel('total_fare_win')
plt.show()

# Plot for base_fare_win
plt.figure(figsize=(8, 4))
sns.boxplot(data=train_df, x='base_fare_win', color='lightgreen')
plt.title('Boxplot: Winsorized Base Fare')
plt.xlabel('base_fare_win')
plt.show()

train_df['total_fare_log_win'] = winsorize(train_df['total_fare_log'], limits=[0.03, 0.01])
train_df['base_fare_log_win'] = winsorize(train_df['base_fare_log'], limits=[0.03, 0.01])

# train_df.columns

# MODEL BUILDING: RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Define features (X) and target (y)
if 'total_fare' in train_df.columns:
    X = train_df.drop(columns=[
        'total_fare', 'total_fare_log', 'total_fare_win', 'total_fare_log_win',
        'base_fare_win', 'base_fare_log', 'base_fare_log_win',
        'booking_date', 'journey_date', 'brd_date', 'trnno',
        'service_charge', 'reservation_fee'
        ], errors='ignore')

    X = X.select_dtypes(include=np.number)
    y = train_df['total_fare'] # Target variable

    # Step 2: Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Output the shape of the splits
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
else:
    print("Error: 'total_fare' column not found in the dataframe.")

from sklearn.metrics import mean_squared_error, r2_score

# Predictions on training set
y_train_pred = model.predict(X_train)

# Training performance
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Train MSE: {train_mse:.2f}")
print(f"Train R2: {train_r2:.2f}")

# Already computed for test:
print(f"Test MSE: {mse:.2f}")
print(f"Test R2: {r2:.2f}")


# HYPERPARAMETRIZED TUNING
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2']
}

model = RandomForestRegressor(random_state=42, n_jobs=-1)
rand_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                 n_iter=10, cv=3, scoring='r2', n_jobs=-1)
X_small = X_train.sample(50000, random_state=42)
y_small = y_train.loc[X_small.index]

rand_search.fit(X_small, y_small)

best_params = rand_search.best_params_

final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    n_jobs=-1,
    random_state=42
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print("R2-score:", r2_score(y_test, y_pred))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# DYNAMIC PRICE LOGIC (Random Forest)
def dynamic_price_predictor(input_data: dict, model, feature_columns, demand_threshold=0.8, surge_rate=0.1):
    import pandas as pd
    import numpy as np

    # Step 1: Convert dict to DataFrame
    input_df = pd.DataFrame([input_data])

    # Step 2: Align input with training features
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing features with 0 (if one-hot encoded or not applicable)

    input_df = input_df[feature_columns]  # Ensure correct column order

    # Step 3: Predict base fare
    predicted_fare = model.predict(input_df)[0]

    # Step 4: Demand-based adjustment (if seat_availability is present)
    if 'seats_left' in input_data and 'total_seats' in input_data:
        seats_left = input_data['seats_left']
        total_seats = input_data['total_seats']
        demand_ratio = 1 - (seats_left / total_seats)

        if demand_ratio >= demand_threshold:
            predicted_fare *= (1 + surge_rate)

    return round(predicted_fare, 2)

# Assume this is a new user booking request
user_input = {
    'booking_lead_time': 12,
    'base_fare': 400,
    'is_weekend': 1,
    'is_holiday': 0,
    'distance': 600,
    'travel_class_3A': 1,
    'travel_class_2A': 0,
    'train_type_Express': 1,
    'train_type_Superfast': 0,
    'seats_left': 30,
    'total_seats': 100,
    # Add any other numerical or one-hot encoded features used
}

# Predict dynamic fare
fare = dynamic_price_predictor(
    input_data=user_input,
    model=final_model,
    feature_columns=X_train.columns.tolist(),
    demand_threshold=0.8,    # Demand threshold (e.g., 80% booked)
    surge_rate=0.1           # 10% price hike if high demand
)

print(f"Dynamic Fare: ₹{fare}")

print("This prices looks accurate.")

print("Lets apply **XG Boost model** on target_fare_log_win for accurate prediction")

# MODEL BUILDING: XGBOOST
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

# Step 1: select features and target
drop_cols = [
    'booking_date', 'journey_date', 'brd_date',
    'total_fare', 'total_fare_win', 'total_fare_log',
    'base_fare_log', 'base_fare_win', 'base_fare_log_win','total_fare_log_win'
]
features = train_df.drop(columns=drop_cols, errors='ignore')
target = train_df['total_fare_log_win']

# Step 2: Train-test split
X_train_xgb, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Step 3: Train XG Boost Model
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_xgb, y_train)

# Step 4: Predict and evaluate
y_pred = xgb_model.predict(X_test)

# Convert back to original fare scale
y_pred_final = np.expm1(y_pred)
y_true = np.expm1(y_test)

# metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred_final))
r2 = r2_score(y_true, y_pred_final)

print("RMSE (actual scale):", round(rmse, 2))
print("R² Score (actual scale):", round(r2, 4))

print("Let's apply hyperparametirzed tuning.")

# HYPERPARAMETRIZED TUNING (XGBoost)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

rs = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=1,
    scoring='neg_root_mean_squared_error'
)

rs.fit(X_train_xgb, y_train)

best_model = rs.best_estimator_

# Train the best model on the training data
best_model.fit(X_train_xgb, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Convert back to original fare scale
y_pred_final = np.expm1(y_pred)
y_true = np.expm1(y_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_true, y_pred_final))
r2 = r2_score(y_true, y_pred_final)

print("Tuned XGBoost RMSE (actual scale):", round(rmse, 2))
print("Tuned XGBoost R² Score (actual scale):", round(r2, 4))


# DYNAMIC PRICE LOGIC (XGBoost)
import numpy as np
import pandas as pd

def dynamic_price_predictor(input_data, model, feature_columns, demand_threshold=0.8, surge_rate=0.1):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict log(fare) and convert back
    log_pred = model.predict(input_df)[0]
    fare = np.expm1(log_pred)

    # Apply surge pricing if demand is high
    seats_left = input_data.get("seats_left", 0)
    total_seats = input_data.get("total_seats", 1)
    demand_ratio = 1 - (seats_left / total_seats)

    if demand_ratio > demand_threshold:
        fare *= (1 + surge_rate)

    # Ensure fare is not below base fare
    base_fare = input_data.get("base_fare", 0)
    final_fare = max(fare, base_fare)

    return round(final_fare, 2)

# Assume this is a new user booking request
user_input = {
    'booking_lead_time': 12,
    'base_fare': 400,
    'is_weekend': 1,
    'is_holiday': 0,
    'distance': 600,
    'travel_class_3A': 1,
    'travel_class_2A': 0,
    'train_type_Express': 1,
    'train_type_Superfast': 0,
    'seats_left': 30,
    'total_seats': 100,
    # Add any other numerical or one-hot encoded features used
}

# Predict dynamic fare
fare = dynamic_price_predictor(
    input_data=user_input,
    model=xgb_model,
    feature_columns=X_train_xgb.columns.tolist(),
    demand_threshold=0.8,    # Demand threshold (e.g., 80% booked)
    surge_rate=0.1           # 10% price hike if high demand
)
print(f"Dynamic Fare: ₹{fare}")
print("#RMSE and R2 socre for tuned XG Boost are efficient.")


# MODEL ENSEMBLING (RF + XGB)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# === Step 1: Feature Engineering (for ensemble)
train_df['booking_lead_time_log'] = np.log1p(train_df['booking_lead_time'])
train_df['train_month_demand'] = train_df['journey_month'] * train_df['train_demand_score']

# === Step 2: Define consistent drop columns
drop_cols = [
    'booking_date', 'journey_date', 'brd_date', 'resupto_date',
    'total_fare', 'total_fare_win', 'total_fare_log',
    'base_fare_log', 'base_fare_win', 'base_fare_log_win',
    'target_fare', 'target_fare_log_win',
    'total_fare_log_win'  # this is target
]

# === Step 3: Create feature matrix and target
X = train_df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=np.number)  # only numeric features

y = train_df['total_fare_log_win']

# === Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 5: Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# === Step 6: Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Step 7: Predict and Average
y_pred_rf_log = rf_model.predict(X_test)
y_pred_xgb_log = xgb_model.predict(X_test)
y_pred_avg_log = (y_pred_rf_log + y_pred_xgb_log) / 2
            
# Step 8: Convert to original scale
y_test_actual = np.expm1(y_test)
y_pred_avg = np.expm1(y_pred_avg_log)

#Step 9: Evaluate
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_avg))
r2 = r2_score(y_test_actual, y_pred_avg)
print("Ensemble (Average of RF + XGB)")
print(f"RMSE (actual scale): ₹{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

print("#The ensembled model is more accurate, so this is the final model.")

def dynamic_price_predictor_ensemble(input_data, rf_model, xgb_model, feature_columns, demand_threshold=0.8, surge_rate=0.1):
    import numpy as np
    import pandas as pd
    # Convert input to DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict log(fare) from both models
    log_pred_rf = rf_model.predict(input_df)[0]
    log_pred_xgb = xgb_model.predict(input_df)[0]
    # Average the predictions and inverse transform
    avg_log_pred = (log_pred_rf + log_pred_xgb) / 2
    fare = np.expm1(avg_log_pred)
    # Apply surge pricing based on demand
    seats_left = input_data.get("seats_left", 0)
    total_seats = input_data.get("total_seats", 1)  # prevent divide-by-zero
    demand_ratio = 1 - (seats_left / total_seats)

    if demand_ratio > demand_threshold:
        fare *= (1 + surge_rate)

    # Ensure fare is not below base fare
    base_fare = input_data.get("base_fare", 0)
    final_fare = max(fare, base_fare)
    return round(final_fare, 2)

# Example user input
user_input = {
    'booking_lead_time': 12,
    'base_fare': 400,
    'is_weekend': 1,
    'is_holiday': 0,
    'distance': 600,
    'travel_class_3A': 1,
    'travel_class_2A': 0,
    'train_type_Express': 1,
    'train_type_Superfast': 0,
    'seats_left': 30,
    'total_seats': 100,
    # add any other required features (numerical or one-hot encoded)
}

# Call the ensemble predictor
fare = dynamic_price_predictor_ensemble(
    input_data=user_input,
    rf_model=rf_model,
    xgb_model=xgb_model,
    feature_columns=X_train.columns.tolist(),  # from your model training
    demand_threshold=0.8,
    surge_rate=0.1
)
print(f"Dynamic Fare (Ensemble): ₹{fare}")

# FEATURE IMPORTANCE PLOTS
from xgboost import plot_importance
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, height=0.5, max_num_features=20)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

import shap
import matplotlib.pyplot as plt
# TreeExplainer works well for tree models like XGBoost and is CPU-safe
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
