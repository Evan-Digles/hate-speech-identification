"""
Quick test script to verify all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing dependency imports...\n")
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tensorflow", "tensorflow"),
        ("tf_keras", "tf_keras"),
        ("torch", "torch"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("nltk", "nltk"),
        ("transformers", "transformers"),
        ("emoji", "emoji"),
        ("tkinter (GUI)", "tkinter")
    ]
    
    failed = []
    
    for display_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"✗ {display_name} - FAILED")
            failed.append(display_name)
    
    if failed:
        print(f"\n✗ Missing dependencies: {', '.join(failed)}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed successfully!")
        print("\nYou're ready to use the Hate Speech Classifier!")
        print("Run: python EvanDiglesCOSC670FinalProject.py")
        return True

def main():
    """Run dependency check"""
    print("=" * 70)
    print("Hate Speech Classifier - Dependency Check")
    print("=" * 70)
    print()
    
    success = test_imports()
    
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
