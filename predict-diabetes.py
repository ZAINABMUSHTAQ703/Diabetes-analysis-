import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
print("🔹 Loading dataset...")
data = pd.read_csv("diabetes.csv")

# 2. Dataset Cleaning & Preprocessing
print("🔹 Cleaning dataset...")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values (if any)
data = data.dropna()

# Encode categorical features
categorical_cols = ["gender", "smoking_history"]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Define features and target
features = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
X = data[features]
y = data["diabetes"]

# 3. Split dataset
print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree Model
print("🔹 Training model...")
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 8. Decision Tree Visualization
plt.figure(figsize=(15,8))
plot_tree(model, feature_names=features, class_names=["No Diabetes", "Diabetes"], filled=True, rounded=True)
plt.show()

# 9. Feature Importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(8,5), color="teal")
plt.title("Feature Importance in Diabetes Prediction")
plt.show()

# 10. User Input Prediction
print("\n🔹 Predict Diabetes from User Input:")
try:
    age = int(input("Enter Age: "))
    bmi = float(input("Enter BMI: "))
    glucose = float(input("Enter Blood Glucose Level: "))
    hba1c = float(input("Enter HbA1c Level: "))
    hypertension = input("Do you have Hypertension? (yes/no): ").lower()
    heart_disease = input("Do you have Heart Disease? (yes/no): ").lower()

    hypertension = 1 if hypertension == "yes" else 0
    heart_disease = 1 if heart_disease == "yes" else 0

    user_data = pd.DataFrame([[age, hypertension, heart_disease, bmi, hba1c, glucose]], 
                              columns=features)
    prediction = model.predict(user_data)[0]
    print("✅ Result:", "Diabetic" if prediction == 1 else "Not Diabetic")
except Exception as e:
    print("⚠️ Error in input:", e)
