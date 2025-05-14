import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("step_data2.csv")

# Encode categorical nodes
df['start_code'] = df['start'].astype('category').cat.codes
df['goal_code'] = df['goal'].astype('category').cat.codes
df['next_code'] = df['next_preferred_node'].astype('category').cat.codes

# Save label mappings
joblib.dump(dict(enumerate(df['next_preferred_node'].astype('category').cat.categories)), "next_node_map.pkl")

# Features and target
X = df[['start_code', 'goal_code', 'distance', 'turns']]
y = df['next_code']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save model
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "step_model.pkl")
