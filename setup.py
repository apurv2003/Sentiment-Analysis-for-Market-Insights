"""
Setup script for Sentiment Analysis Project
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 SENTIMENT ANALYSIS PROJECT SETUP")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️ Warning: Could not download spaCy model. Some features may not work.")
    
    # Download NLTK data
    nltk_script = """
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
print("NLTK data downloaded successfully!")
"""
    
    if not run_command(f'python -c "{nltk_script}"', "Downloading NLTK data"):
        print("⚠️ Warning: Could not download NLTK data. Some features may not work.")
    
    # Create necessary directories
    directories = ['data', 'models', 'results', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Run the training pipeline:")
    print("   python src/train_models.py")
    print("\n2. Start the Streamlit dashboard:")
    print("   streamlit run dashboard/app.py")
    print("\n3. Explore the Jupyter notebook:")
    print("   jupyter notebook notebooks/sentiment_analysis_exploration.ipynb")
    print("\n4. Evaluate models:")
    print("   python src/evaluate_models.py")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage instructions")
    print("- src/: Source code with detailed comments")
    print("- notebooks/: Jupyter notebooks for exploration")
    
    print("\n🔧 Troubleshooting:")
    print("- If you encounter GPU issues, models will automatically use CPU")
    print("- For large datasets, consider using cloud computing resources")
    print("- Check the logs in results/ directory for detailed information")
    
    print("\n✅ You're all set! Happy sentiment analyzing! 🎯")

if __name__ == "__main__":
    main() 