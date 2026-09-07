import os
import joblib
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_tune_models, MODELS_DIR
from src.evaluator import evaluate_all_models
from src.generate_report import generate_html_report
from src.predict import predict_patient_risk

def main():
    print("\n" + "="*70)
    print(" [HealthPredict AI] - End-to-End Machine Learning Pipeline")
    print("="*70)

    # 1. Load & Preprocess Data
    print("\n[Step 1/5] Loading and standardizing clinical patient cohort...")
    (
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, feature_cols, raw_df
    ) = load_and_preprocess_data(test_size=0.2, random_state=42)

    # Save Scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Standard scaler artifact saved to: {scaler_path}")

    # 2. Train & Tune Benchmark Models
    print("\n[Step 2/5] Benchmarking classification models & fine-tuning hyperparameters...")
    results, best_model = train_and_tune_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # 3. Comprehensive Model Evaluation & Diagnostic Visualizations
    print("\n[Step 3/5] Calculating evaluation metrics and generating diagnostic figures...")
    metrics_df = evaluate_all_models(results, y_test, feature_cols)

    # 4. Generate Interactive Clinical Dashboard
    print("\n[Step 4/5] Compiling HTML clinical risk analytics dashboard...")
    dataset_summary = {
        "n_patients": f"{len(raw_df):,}",
        "features": len(feature_cols)
    }
    report_path = generate_html_report(metrics_df, dataset_summary)

    # 5. Test Live Sample Inference
    print("\n[Step 5/5] Testing patient risk inference engine with a live profile...")
    sample_patient = {
        "age": 62,
        "gender": 1,
        "systolic_bp": 152,
        "diastolic_bp": 96,
        "resting_heart_rate": 84,
        "cholesterol_total": 260,
        "hdl_cholesterol": 36,
        "ldl_cholesterol": 178,
        "fasting_glucose": 142,
        "bmi": 32.8,
        "smoking_status": 2,
        "physical_activity_hours": 0.5,
        "family_history": 1
    }
    risk_assessment = predict_patient_risk(sample_patient)
    print("\n--- Clinical Risk Triage Output ---")
    print(f"Prediction: {'HIGH RISK' if risk_assessment['prediction'] == 1 else 'LOW/MODERATE RISK'}")
    print(f"Disease Probability: {risk_assessment['disease_probability_percent']}")
    print(f"Risk Tier: {risk_assessment['risk_tier']}")
    print(f"Primary Risk Factors: {', '.join(risk_assessment['primary_risk_factors'])}")
    print(f"Actionable Guidance: {risk_assessment['clinical_recommendation']}")

    print("\n" + "="*70)
    print(" [OK] HealthPredict AI Pipeline Completed Successfully!")
    print(f" [INFO] Open the dashboard report at: {report_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
