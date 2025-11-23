import pandas as pd
import numpy as np

N = 50000

np.random.seed(42)
data = {
    "transaction_id": np.arange(1, N+1),
    "user_id": np.random.randint(1, 2000, N),
    "amount": np.round(np.random.uniform(5, 2000, N), 2),
    "merchant": np.random.choice(["Amazon", "MercadoLivre", "Shopee", "iFood", "Steam", "Netflix", "Uber"], N),
    "country": np.random.choice(["BR", "US", "IN", "CN", "FR", "DE"], N),
    "timestamp": pd.date_range("2024-01-01", periods=N, freq="min"),
    "is_fraud": np.random.choice([0, 1], N, p=[0.97, 0.03])
}

df = pd.DataFrame(data)
df.to_csv("transactions_all.csv", index=False)
print("✅ Dataset gerado: transactions_all.csv")


import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("transactions_all.csv")

X = df[["amount", "merchant", "country"]]
y = df["is_fraud"]

hasher = FeatureHasher(n_features=16, input_type="string")

X_hashed = hasher.transform(
    X[["merchant", "country"]].astype(str).to_dict(orient="records")
)
X_final = np.hstack([X["amount"].values.reshape(-1, 1), X_hashed.toarray()])

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_model.pkl")
joblib.dump(hasher, "feature_hasher.pkl")
print("✅ Modelo salvo!")

import pandas as pd
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
hasher = joblib.load("feature_hasher.pkl")

print("🧠 Sistema de Análise de Risco de Fraude\n")

amount = float(input("💰 Digite o valor da compra (R$): "))
merchant = input("🏪 Digite o nome do comerciante (ex: Amazon, Shopee, iFood): ")
country = input("🌎 Digite o país (ex: BR, US, CN): ")

new_data = pd.DataFrame({
    "amount": [amount],
    "merchant": [merchant],
    "country": [country]
})

X_hashed = hasher.transform(new_data[["merchant", "country"]].astype(str).to_dict(orient="records"))
X_final = np.hstack([new_data["amount"].values.reshape(-1, 1), X_hashed.toarray()])

fraud_prob = model.predict_proba(X_final)[0][1] * 100

print(f"\n🔎 Probabilidade estimada de fraude: {fraud_prob:.2f}%")

if fraud_prob < 30:
    print("🟢 Compra parece NORMAL (baixo risco).")
elif 30 <= fraud_prob < 70:
    print("🟡 Compra SUSPEITA — verifique se foi você mesmo.")
else:
    print("🔴 ALERTA: Alta chance de FRAUDE ou CLONAGEM!")



    
