# ================================================================
# train_model.py
# Human Fatigue Prediction — Model Training Script
# ================================================================
# Cara menjalankan:
#   cd human_fatigue_project
#   python train_model.py
# ================================================================

import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')


# ================================================================
# STEP 1 — KONFIGURASI PATH
# ================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'human_fatigue.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# STEP 2 — LOAD DATA
# ================================================================

def load_data(path):
    print("\n" + "="*55)
    print("📂 STEP 1 — LOAD DATA")
    print("="*55)

    df = pd.read_csv(path)

    print(f"✅ Dataset loaded     : {path}")
    print(f"✅ Shape              : {df.shape}")
    print(f"✅ Kolom              : {list(df.columns)}")

    return df


# ================================================================
# STEP 3 — DATA CLEANING
# ================================================================

def clean_data(df):
    print("\n" + "="*55)
    print("🧹 STEP 2 — DATA CLEANING")
    print("="*55)

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"✅ Hapus duplikat     : {before - after} baris dihapus")

    # Drop kolom data leakage
    cols_to_drop = ['System_Recommendation', 'Decision_Fatigue_Score']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"✅ Drop kolom leakage : {existing_drops}")
    print(f"✅ Kolom tersisa      : {list(df.columns)}")
    print(f"✅ Missing values     : {df.isnull().sum().sum()} total")

    return df


# ================================================================
# STEP 4 — PREPROCESSING & SPLIT
# ================================================================

def preprocess(df):
    print("\n" + "="*55)
    print("⚙️  STEP 3 — PREPROCESSING")
    print("="*55)

    # Pisahkan fitur & target
    X = df.drop(columns=['Fatigue_Level'])
    y = df['Fatigue_Level']

    # Label encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"✅ Target classes     : {list(label_encoder.classes_)}")
    print(f"✅ Target mapping     : "
          f"{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Definisi fitur
    numeric_features = [
        'Hours_Awake',
        'Decisions_Made',
        'Task_Switches',
        'Avg_Decision_Time_sec',
        'Sleep_Hours_Last_Night',
        'Caffeine_Intake_Cups',
        'Stress_Level_1_10',
        'Error_Rate',
        'Cognitive_Load_Score'
    ]

    categorical_features = ['Time_of_Day']

    # Pastikan semua kolom ada
    for col in numeric_features + categorical_features:
        if col not in X.columns:
            raise ValueError(f"❌ Kolom '{col}' tidak ditemukan di dataset!")

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"✅ Train size         : {X_train.shape}")
    print(f"✅ Test size          : {X_test.shape}")

    return (
        X, X_train, X_test,
        y_train, y_test,
        preprocessor, label_encoder,
        numeric_features, categorical_features
    )


# ================================================================
# STEP 5 — TRAINING & EVALUASI
# ================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       preprocessor, label_encoder):

    print("\n" + "="*55)
    print("🤖 STEP 4 — TRAINING & EVALUASI")
    print("="*55)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}

    for name, clf in models.items():
        # Buat pipeline per model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        results[name] = {
            'pipeline'  : pipeline,
            'accuracy'  : acc,
            'y_pred'    : y_pred
        }

        print(f"\n{'─'*40}")
        print(f"📊 {name}")
        print(f"{'─'*40}")
        print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

    # Pilih model terbaik
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_acc  = results[best_name]['accuracy']

    print(f"\n{'='*55}")
    print(f"🏆 Model Terbaik : {best_name}")
    print(f"🏆 Accuracy      : {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"{'='*55}")

    return results, best_name


# ================================================================
# STEP 6 — SAVE MODEL
# ================================================================

def save_model(results, best_name, label_encoder, X, MODEL_DIR):
    print("\n" + "="*55)
    print("💾 STEP 5 — SAVE MODEL")
    print("="*55)

    rf_pipeline  = results['Random Forest']['pipeline']
    model_path   = os.path.join(MODEL_DIR, 'fatigue_model.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    columns_path = os.path.join(MODEL_DIR, 'columns.pkl')

    joblib.dump(rf_pipeline,     model_path)
    joblib.dump(label_encoder,   encoder_path)
    joblib.dump(list(X.columns), columns_path)

    # ✅ SIMPAN ACCURACY SEBAGAI FLOAT BUKAN DICT BERSARANG
    accuracy_data = {
        name: float(round(res['accuracy'] * 100, 2))
        for name, res in results.items()
    }
    accuracy_path = os.path.join(MODEL_DIR, 'model_accuracy.pkl')
    joblib.dump(accuracy_data, accuracy_path)

    print(f"✅ Model saved     : {model_path}")
    print(f"✅ Encoder saved   : {encoder_path}")
    print(f"✅ Columns saved   : {columns_path}")
    print(f"✅ Accuracy saved  : {accuracy_path}")
    print(f"\n📊 Accuracy per model:")
    for name, acc in accuracy_data.items():
        print(f"   {name:25s}: {acc:.2f}%")

# ================================================================
# MAIN — JALANKAN SEMUA STEP
# ================================================================

def main():
    print("\n" + "🔥"*20)
    print("  HUMAN FATIGUE PREDICTION — MODEL TRAINING")
    print("🔥"*20)

    # Jalankan semua step
    df                      = load_data(DATA_PATH)
    df                      = clean_data(df)
    (X, X_train, X_test,
     y_train, y_test,
     preprocessor,
     label_encoder,
     numeric_features,
     categorical_features)  = preprocess(df)

    results, best_name      = train_and_evaluate(
                                X_train, X_test,
                                y_train, y_test,
                                preprocessor, label_encoder
                              )

    save_model(results, best_name, label_encoder, X, MODEL_DIR)

    print("\n" + "✅"*20)
    print("  TRAINING SELESAI! Model siap digunakan.")
    print("  Jalankan: streamlit run app.py")
    print("✅"*20 + "\n")


if __name__ == '__main__':
    main()