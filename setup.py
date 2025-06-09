import subprocess
import sys

def setup_environment():
    print("Setting up Mood Detection System...")
    
    # Required packages
    packages = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'networkx>=2.6.0',
        'jupyter>=1.0.0',
        'notebook>=6.4.0'
    ]
    
    print("\nInstalling required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nSetup completed successfully!")
    print("\nTo start using the system:")
    print("1. Run 'python test_mood_detection.py' to test the system")
    print("2. Run 'jupyter notebook' to open the analysis notebook")
    print("3. Open 'mood_detection_analysis.ipynb' in Jupyter")

if __name__ == "__main__":
    setup_environment() 