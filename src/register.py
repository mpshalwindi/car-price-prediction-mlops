
import argparse
from pathlib import Path
from azureml.core import Model, Run
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str)
    args = parser.parse_args()
    
    print("🚀 STARTING ROBUST MODEL REGISTRATION")
    print(f"Model input path: {args.model_input}")
    
    # Get the run context
    run = Run.get_context()
    print("✅ Azure ML Run context acquired")
    
    # The model input is a FOLDER, we need to find the model file inside it
    model_dir = Path(args.model_input)
    print(f"🔍 Searching for model in directory: {model_dir}")
    
    if not model_dir.exists():
        print(f"❌ Model directory does not exist: {model_dir}")
        return
    
    # List all files in the model directory
    all_files = list(model_dir.iterdir())
    print(f"📁 Found {len(all_files)} files in model directory:")
    for file in all_files:
        print(f"   - {file.name} (size: {file.stat().st_size if file.is_file() else 'dir'} bytes)")
    
    # Look for the actual model file
    model_file = None
    for file in all_files:
        if file.is_file() and file.name == "model.joblib":
            model_file = file
            break
    
    if not model_file:
        print("❌ model.joblib file not found in the directory")
        # Try to find any model file
        for file in all_files:
            if file.is_file() and ('.joblib' in file.suffix or '.pkl' in file.suffix):
                model_file = file
                print(f"🔄 Found alternative model file: {file.name}")
                break
    
    if model_file:
        print(f"✅ Found model file: {model_file}")
        print(f"📏 Model file size: {model_file.stat().st_size} bytes")
        
        # CRITICAL: Upload the model file to the run's outputs
        # This makes it available for registration
        output_model_path = "outputs/model.joblib"
        run.upload_file(output_model_path, str(model_file))
        print(f"✅ Model file uploaded to: {output_model_path}")
        
        # Wait a moment for upload to complete
        import time
        time.sleep(2)
        
        # Now register from the uploaded location
        try:
            model = run.register_model(
                model_name='used-car-price-predictor',
                model_path=output_model_path,  # Use the uploaded path
                description='Random Forest model for used car price prediction - Fixed Pipeline',
                tags={
                    'framework': 'scikit-learn', 
                    'type': 'regression',
                    'algorithm': 'random_forest',
                    'pipeline': 'used-car-price-e2e'
                }
            )
            print(f"🎉 MODEL REGISTERED SUCCESSFULLY!")
            print(f"   Name: {model.name}")
            print(f"   Version: {model.version}")
            print(f"   ID: {model.id}")
            
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            print("🔄 Trying alternative registration method...")
            
            # Alternative: Copy to a known location and register
            try:
                local_model_path = "/tmp/model.joblib"
                shutil.copy2(str(model_file), local_model_path)
                model = Model.register(
                    workspace=run.experiment.workspace,
                    model_path=local_model_path,
                    model_name='used-car-price-predictor',
                    description='Random Forest model - Alternative registration',
                    tags={'method': 'alternative'}
                )
                print(f"✅ Alternative registration successful: {model.name} v{model.version}")
            except Exception as e2:
                print(f"❌ Alternative registration also failed: {e2}")
                
    else:
        print("❌ No model file found at all!")
        print("   Expected: model.joblib or any .joblib/.pkl file")
        
        # Create a dummy file for testing if needed
        print("🔄 Creating test file for debugging...")
        test_file = model_dir / "test_debug.txt"
        test_file.write_text("This is a test file for debugging")
        print(f"✅ Created test file: {test_file}")
    
    print("🎯 REGISTRATION PROCESS COMPLETED")

if __name__ == "__main__":
    main()
