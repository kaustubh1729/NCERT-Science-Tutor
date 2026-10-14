#!/usr/bin/env python3
"""
Setup script for Render deployment
Handles initial data setup and database creation
"""

import os
import sys
import warnings

# Silently ignore transformers deprecation warnings
os.environ['TRANSFORMERS_NO_ADVISORY'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__` from.*")
warnings.filterwarnings("ignore", message=".*This alias will be removed.*")

from pathlib import Path

def check_data_files():
    """Check if NCERT PDF files are present"""
    data_dir = Path("data")
    required_classes = [6, 7, 8, 9, 10]
    
    print("📚 Checking NCERT data files...")
    
    missing_files = []
    for class_num in required_classes:
        class_dir = data_dir / f"class_{class_num}"
        pdf_file = class_dir / f"ncert_class{class_num}_science.pdf"
        
        if not pdf_file.exists():
            missing_files.append(str(pdf_file))
            print(f"  ❌ Missing: {pdf_file}")
        else:
            print(f"  ✅ Found: {pdf_file}")
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} PDF files missing!")
        print("   The application will not work without NCERT textbooks.")
        print("   Please ensure all PDF files are in the data/ directory.")
        return False
    
    print("\n✅ All NCERT data files present!")
    return True


def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = ["data", "db", "cache", "logs"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✅ Created/verified: {dir_name}/")


def check_env_variables():
    """Check if required environment variables are set"""
    print("\n🔑 Checking environment variables...")
    
    required_vars = ["COHERE_API_KEY", "NVIDIA_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"  ❌ Missing: {var}")
        else:
            print(f"  ✅ Found: {var}")
    
    if missing_vars:
        print(f"\n⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your Render dashboard.")
        return False
    
    print("\n✅ All environment variables configured!")
    return True


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = [
        "streamlit",
        "chromadb",
        "cohere",
        "langchain",
        "sentence_transformers",
        "pymupdf",
        "rank_bm25",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (not installed)")
    
    if missing_packages:
        print(f"\n❌ Error: Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def create_vector_database():
    """Create vector database if it doesn't exist"""
    print("\n🗄️  Checking vector database...")
    
    db_path = Path("db")
    if db_path.exists() and any(db_path.iterdir()):
        print("  ℹ️  Vector database already exists")
        return True
    
    print("  🔄 Creating vector database (this may take a few minutes)...")
    
    try:
        from src.pipeline.rag_pipeline import RAGPipeline
        from dotenv import load_dotenv
        
        load_dotenv()
        cohere_key = os.getenv("COHERE_API_KEY")
        
        if not cohere_key:
            print("  ❌ COHERE_API_KEY not set. Cannot create vector database.")
            return False
        
        print("  📊 Loading and processing NCERT documents...")
        pipeline = RAGPipeline(cohere_api_key=cohere_key)
        
        print("  ✅ Vector database created successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating vector database: {e}")
        print("   The app will attempt to create it on first run.")
        return False


def verify_setup():
    """Run all setup checks"""
    print("=" * 80)
    print("  🚀 NCERT AI Tutor - Render Deployment Setup")
    print("=" * 80)
    
    results = {
        "Directories": setup_directories(),
        "Dependencies": check_dependencies(),
        "Environment": check_env_variables(),
        "Data Files": check_data_files(),
        "Vector DB": create_vector_database(),
    }
    
    print("\n" + "=" * 80)
    print("  📋 Setup Summary")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("  ✅ Setup Complete! Your app is ready to deploy on Render.")
        print("=" * 80)
        print("\n📝 Next Steps:")
        print("  1. Push your code to GitHub")
        print("  2. Connect your repository to Render")
        print("  3. Set environment variables in Render dashboard")
        print("  4. Deploy!")
        return 0
    else:
        print("  ⚠️  Setup incomplete. Please resolve the issues above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    sys.exit(verify_setup())