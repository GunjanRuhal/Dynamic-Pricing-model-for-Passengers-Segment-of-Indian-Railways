# Dynamic Pricing Model for Indian Railways Passenger Segment

This project implements a data science and machine learning pipeline to develop a **two-sided Dynamic Pricing Model** for Indian Railways' passenger segment. The goal is to improve upon the existing rigid "Flexi-fare scheme" by incorporating demand- and time-based factors that allow for both fare hikes and discounts.

## 1. Problem Statement

The existing "Flexi-fare scheme" adopted by Indian Railways primarily functions as a **one-sided fare hike mechanism**. This approach increased revenue but also led to a fall in passenger traffic because:
* **No Downside:** Fares did not decrease when demand was low.
* **Lack of Flexibility:** The uniform application failed to account for varying demand elasticity across different routes and times.

---

## 2. Methodology & Model Components

The code addresses the problem by using features related to time and demand to predict the **Total Fare**, utilizing a pipeline of: Data Processing $\rightarrow$ Feature Engineering $\rightarrow$ Outlier Treatment $\rightarrow$ Ensemble Modeling.

### Feature Engineering Highlights

The script created several new features critical for the model:
* **Temporal Features:** `booking_lead_time`, `is_weekend`, `season`, `journey_month`, `journey_week`.
* **Demand/Contextual Features:** `is_holiday`, `train_demand_score` (based on train frequency/popularity).
* **Interaction Features:** Terms like `lead_time_x_is_holiday` and `is_holiday_and_weekend`.
* **Data Transformation:** The target variable (`total_fare`) was treated using **Log Transformation ($\log(1+x)$)** and **Winsorization** to mitigate the influence of outliers. The final target for the models was the transformed variable, `total_fare_log_win`.

### Model Training and Final Predictor

The final price prediction is driven by an **Ensemble Model** combining two robust tree-based regressors:

1.  **Random Forest Regressor**
2.  **XGBoost Regressor (Tuned)**

The **Ensemble Model (RF + XGBoost Average)** achieved the best performance

### Dynamic Price Logic (Surge & Discount)

The final price calculation is performed by the `dynamic_price_predictor_ensemble` function, which implements the two-sided logic:

1.  **Model Prediction:** The average prediction from the trained RF and XGBoost models determines the base price.
2.  **Surge Pricing:** If **demand is high** (e.g., occupancy exceeds a threshold like 80%), the predicted fare is increased by a set surge rate (e.g., 10%).
3.  **Minimum Fare:** The predicted fare is capped to ensure it is **never lower than the original base fare**.

---

## 3. Code Structure

This repository provides the executable pipeline:

* **Data Processing:** Merges booking data and holiday data, performs initial cleanup, and drops redundant columns.
* **Encoding:** Uses **Label Encoding** and **One-Hot Encoding** to prepare categorical features for modeling.
* **Outlier Treatment:** Implements the IQR method for detection, followed by log transformation and Winsorization for feature stabilization.
* **Model Building:** Implements and tunes the RF and XGBoost models, and combines them into the final ensemble.
* **Final Output:** The script concludes with the `dynamic_price_predictor_ensemble` function to demonstrate real-time dynamic pricing capability.
