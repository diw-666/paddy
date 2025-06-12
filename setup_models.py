#!/usr/bin/env python3
"""
Setup script for the Rice Yield Prediction System
This script processes the data and trains the ML models
"""

import os
import sys
from data_processor import RiceDataProcessor
from ml_model import RiceYieldPredictor

def main():
    print("🌾 Setting up Rice Yield Prediction System...")
    print("=" * 50)
    
    # Step 1: Process Data
    print("\n📁 Step 1: Processing data files...")
    processor = RiceDataProcessor()
    
    try:
        # Load and combine all CSV files
        combined_data = processor.load_and_combine_data()
        
        # Create features for machine learning
        print("\n🔧 Creating features for machine learning...")
        features_data = processor.create_features(combined_data)
        
        # Save processed data
        processor.save_processed_data("processed_rice_data.csv")
        
        print(f"✅ Data processing completed!")
        print(f"   - Total records: {len(features_data):,}")
        print(f"   - Districts: {features_data['District'].nunique()}")
        print(f"   - Years: {features_data['Year'].min()} - {features_data['Year'].max()}")
        
    except Exception as e:
        print(f"❌ Error in data processing: {str(e)}")
        return False
    
    # Step 2: Train Models
    print("\n🤖 Step 2: Training machine learning models...")
    predictor = RiceYieldPredictor()
    
    try:
        # Train the models
        predictor.train_models(features_data)
        
        # Save the trained models
        predictor.save_models("rice_yield_models.pkl")
        
        print("✅ Model training completed!")
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False
    
    # Step 3: Test the System
    print("\n🧪 Step 3: Testing the system...")
    
    try:
        # Get average values from the dataset for realistic test
        colombo_data = features_data[features_data['District'] == 'COLOMBO']
        if len(colombo_data) > 0:
            avg_major = int(colombo_data['Major_Schemes_Sown'].mean())
            avg_minor = int(colombo_data['Minor_Schemes_Sown'].mean())
            avg_rainfed = int(colombo_data['Rainfed_Sown'].mean())
        else:
            avg_major = 25000
            avg_minor = 10000
            avg_rainfed = 8000
        
        # Test prediction
        test_prediction = predictor.predict(
            district="COLOMBO",
            season="Yala",
            year=2024,
            Major_Schemes_Sown=avg_major,
            Minor_Schemes_Sown=avg_minor,
            Rainfed_Sown=avg_rainfed,
            Major_Schemes_Harvested=int(avg_major * 0.9),
            Minor_Schemes_Harvested=int(avg_minor * 0.9),
            Rainfed_Harvested=int(avg_rainfed * 0.8),
            All_Schemes_Sown=avg_major + avg_minor + avg_rainfed,
            All_Schemes_Harvested=int((avg_major + avg_minor + avg_rainfed) * 0.88)
        )
        
        print(f"✅ Test prediction successful!")
        print(f"   - District: {test_prediction['district']}")
        print(f"   - Season: {test_prediction['season']} {test_prediction['year']}")
        print(f"   - Predicted Yield: {test_prediction['predicted_yield']:.2f} kg/ha")
        print(f"   - Predicted Production: {test_prediction['predicted_production']:,.0f} tons")
        
        # Test anomaly detection
        anomaly_result = predictor.detect_anomaly(
            district="COLOMBO",
            actual_yield=test_prediction['predicted_yield'] * 0.7,  # Simulate low yield
            actual_production=test_prediction['predicted_production'] * 0.7,
            predicted_yield=test_prediction['predicted_yield'],
            predicted_production=test_prediction['predicted_production']
        )
        
        print(f"\n🔍 Anomaly detection test: {anomaly_result['message']}")
        
    except Exception as e:
        print(f"❌ Error in system testing: {str(e)}")
        return False
    
    # Step 4: Generate Summary
    print("\n📋 Step 4: System Summary")
    print("=" * 50)
    
    # Check if all required files exist
    required_files = [
        "processed_rice_data.csv",
        "rice_yield_models.pkl",
        "data_processor.py",
        "ml_model.py",
        "dashboard.py",
        "requirements.txt"
    ]
    
    print("📂 Generated Files:")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (missing)")
    
    print("\n🚀 Setup completed successfully!")
    print("\nTo run the dashboard:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run dashboard: streamlit run dashboard.py")
    print("\nThe system is ready to use! 🎉")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 