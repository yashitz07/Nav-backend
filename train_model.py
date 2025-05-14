import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import ast

# Load the dataset
df = pd.read_csv("step_data.csv")

# Encode nodes
df['current_code'] = df['current_node'].astype('category').cat.codes
df['target_code'] = df['target_node'].astype('category').cat.codes
df['next_code'] = df['next_node'].astype('category').cat.codes

# Save label mappings
joblib.dump(dict(enumerate(df['next_node'].astype('category').cat.categories)), "next_node_map.pkl")

X = df[['current_code', 'target_code', 'distance', 'turns']]
y = df['next_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Step-wise Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "step_model.pkl")
