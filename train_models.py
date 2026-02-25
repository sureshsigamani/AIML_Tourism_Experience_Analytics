import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import joblib

# Load data
df = pd.read_csv("tourism.csv")

# Encode categorical columns
le_country = LabelEncoder()
le_city = LabelEncoder()
le_attraction = LabelEncoder()
le_visit = LabelEncoder()

df["Country"] = le_country.fit_transform(df["Country"])
df["City"] = le_city.fit_transform(df["City"])
df["Attraction"] = le_attraction.fit_transform(df["Attraction"])
df["VisitModeEncoded"] = le_visit.fit_transform(df["VisitMode"])

# Features
X = df[["Country","City","Attraction","Year","Month"]]

# ---------- Regression (Rating) ----------
y_rating = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

pred_rating = reg.predict(X_test)

print("Rating R2:", r2_score(y_test, pred_rating))

joblib.dump(reg, "rating_model.pkl")

# ---------- Classification (VisitMode) ----------
y_visit = df["VisitModeEncoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y_visit, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pred_visit = clf.predict(X_test)

print("VisitMode Accuracy:", accuracy_score(y_test, pred_visit))

joblib.dump(clf, "visit_model.pkl")
joblib.dump(le_visit, "visit_encoder.pkl")

print("Models saved")