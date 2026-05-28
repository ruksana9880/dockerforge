import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_cardiovascular_dataset(n_samples=10000):
    """
    Generate synthetic cardiovascular disease dataset based on blood test parameters
    """
    np.random.seed(42)
    random.seed(42)
    
    # Age distribution (20-80 years)
    ages = np.random.normal(50, 15, n_samples)
    ages = np.clip(ages, 20, 80)
    
    # Gender (0: Female, 1: Male)
    genders = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])
    
    # Blood pressure (systolic/diastolic)
    # Normal: 120/80, High: 140/90+
    systolic_bp = np.random.normal(125, 20, n_samples)
    diastolic_bp = np.random.normal(82, 12, n_samples)
    
    # Cholesterol levels (mg/dL)
    # Total cholesterol: Normal <200, High >240
    total_cholesterol = np.random.normal(200, 40, n_samples)
    
    # LDL (bad cholesterol): Normal <100, High >160
    ldl_cholesterol = np.random.normal(120, 35, n_samples)
    
    # HDL (good cholesterol): Normal >40 (men), >50 (women)
    hdl_cholesterol = np.where(genders == 1, 
                              np.random.normal(45, 12, n_samples),
                              np.random.normal(55, 15, n_samples))
    
    # Triglycerides: Normal <150, High >200
    triglycerides = np.random.normal(140, 60, n_samples)
    
    # Glucose levels (mg/dL): Normal 70-100, Diabetes >126
    glucose = np.random.normal(95, 25, n_samples)
    
    # Hemoglobin A1c (%): Normal <5.7, Diabetes >6.5
    hba1c = np.random.normal(5.5, 1.2, n_samples)
    
    # C-reactive protein (mg/L): Normal <3, High >10
    crp = np.random.exponential(2, n_samples)
    
    # Troponin (ng/mL): Normal <0.04, Elevated >0.4
    troponin = np.random.exponential(0.02, n_samples)
    
    # BNP (pg/mL): Normal <100, Heart failure >400
    bnp = np.random.exponential(80, n_samples)
    
    # Homocysteine (μmol/L): Normal <15, High >15
    homocysteine = np.random.normal(12, 5, n_samples)
    
    # Smoking status (0: No, 1: Yes)
    smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Family history (0: No, 1: Yes)
    family_history = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # BMI
    bmi = np.random.normal(26, 5, n_samples)
    bmi = np.clip(bmi, 15, 50)
    
    # Create cardiovascular disease target based on risk factors
    risk_score = (
        (ages - 20) / 60 * 0.3 +  # Age factor
        genders * 0.1 +  # Gender factor (males higher risk)
        np.clip((systolic_bp - 120) / 40, 0, 1) * 0.2 +  # BP factor
        np.clip((total_cholesterol - 200) / 100, 0, 1) * 0.15 +  # Cholesterol
        np.clip((glucose - 100) / 50, 0, 1) * 0.1 +  # Glucose
        np.clip(crp / 10, 0, 1) * 0.05 +  # Inflammation
        smoking * 0.15 +  # Smoking
        family_history * 0.1 +  # Family history
        np.clip((bmi - 25) / 15, 0, 1) * 0.1  # BMI
    )
    
    # Add some noise and create binary target
    risk_score += np.random.normal(0, 0.1, n_samples)
    cardiovascular_disease = (risk_score > 0.6).astype(int)
    
    # Adjust some values based on disease status to make it more realistic
    mask_diseased = cardiovascular_disease == 1
    
    # Diseased patients tend to have worse values
    systolic_bp[mask_diseased] += np.random.normal(15, 5, np.sum(mask_diseased))
    diastolic_bp[mask_diseased] += np.random.normal(8, 3, np.sum(mask_diseased))
    total_cholesterol[mask_diseased] += np.random.normal(30, 10, np.sum(mask_diseased))
    ldl_cholesterol[mask_diseased] += np.random.normal(25, 8, np.sum(mask_diseased))
    hdl_cholesterol[mask_diseased] -= np.random.normal(5, 2, np.sum(mask_diseased))
    triglycerides[mask_diseased] += np.random.normal(40, 15, np.sum(mask_diseased))
    glucose[mask_diseased] += np.random.normal(20, 10, np.sum(mask_diseased))
    hba1c[mask_diseased] += np.random.normal(0.8, 0.3, np.sum(mask_diseased))
    crp[mask_diseased] += np.random.exponential(3, np.sum(mask_diseased))
    troponin[mask_diseased] += np.random.exponential(0.1, np.sum(mask_diseased))
    bnp[mask_diseased] += np.random.exponential(150, np.sum(mask_diseased))
    homocysteine[mask_diseased] += np.random.normal(5, 2, np.sum(mask_diseased))
    
    # Ensure realistic ranges
    systolic_bp = np.clip(systolic_bp, 80, 220)
    diastolic_bp = np.clip(diastolic_bp, 50, 150)
    total_cholesterol = np.clip(total_cholesterol, 100, 400)
    ldl_cholesterol = np.clip(ldl_cholesterol, 50, 300)
    hdl_cholesterol = np.clip(hdl_cholesterol, 20, 100)
    triglycerides = np.clip(triglycerides, 50, 500)
    glucose = np.clip(glucose, 60, 300)
    hba1c = np.clip(hba1c, 4.0, 12.0)
    crp = np.clip(crp, 0.1, 50)
    troponin = np.clip(troponin, 0.001, 10)
    bnp = np.clip(bnp, 10, 2000)
    homocysteine = np.clip(homocysteine, 5, 50)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': ages.round(0).astype(int),
        'gender': genders,
        'systolic_bp': systolic_bp.round(0).astype(int),
        'diastolic_bp': diastolic_bp.round(0).astype(int),
        'total_cholesterol': total_cholesterol.round(1),
        'ldl_cholesterol': ldl_cholesterol.round(1),
        'hdl_cholesterol': hdl_cholesterol.round(1),
        'triglycerides': triglycerides.round(1),
        'glucose': glucose.round(1),
        'hba1c': hba1c.round(2),
        'crp': crp.round(2),
        'troponin': troponin.round(3),
        'bnp': bnp.round(1),
        'homocysteine': homocysteine.round(1),
        'smoking': smoking,
        'family_history': family_history,
        'bmi': bmi.round(1),
        'cardiovascular_disease': cardiovascular_disease
    })
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating cardiovascular disease dataset...")
    dataset = generate_cardiovascular_dataset(10000)
    
    # Save to CSV
    dataset.to_csv('cardiovascular_dataset.csv', index=False)
    
    # Display basic statistics
    print(f"\nDataset generated successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Positive cases (CVD): {dataset['cardiovascular_disease'].sum()}")
    print(f"Negative cases (No CVD): {len(dataset) - dataset['cardiovascular_disease'].sum()}")
    print(f"Positive rate: {dataset['cardiovascular_disease'].mean():.2%}")
    
    print("\nDataset preview:")
    print(dataset.head())
    
    print("\nDataset info:")
    print(dataset.info())
    
    print("\nDataset saved as 'cardiovascular_dataset.csv'")