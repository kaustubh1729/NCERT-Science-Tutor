#!/usr/bin/env python3
"""
Demo script for NCERT AI Tutor
Shows example queries and their answers
"""

import os
import sys
import warnings

# Silently ignore transformers deprecation warnings
os.environ['TRANSFORMERS_NO_ADVISORY'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing `__path__` from.*")
warnings.filterwarnings("ignore", message=".*This alias will be removed.*")

from dotenv import load_dotenv
from src.pipeline.rag_pipeline import RAGPipeline

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

load_dotenv()


def print_separator(char="=", length=80):
    print(char * length)


def print_section(title, char="="):
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def run_demo():
    """Run demo queries to showcase the RAG system"""
    
    print_separator()
    print("  📘 NCERT AI Tutor - Demo")
    print_separator()
    
    # Check API keys
    cohere_key = os.getenv("COHERE_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    
    if not cohere_key or not nvidia_key:
        print("\n❌ Error: API keys not found!")
        print("Please set COHERE_API_KEY and NVIDIA_API_KEY in .env file\n")
        return
    
    print("\n✅ API keys configured")
    print("🔧 Initializing RAG Pipeline...")
    
    try:
        pipeline = RAGPipeline(cohere_api_key=cohere_key)
        print("✅ Pipeline initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}\n")
        return
    
    # Demo queries
    demo_queries = [
        "What is photosynthesis?",
        "Explain Newton's first law of motion",
        "What are the characteristics of mammals?",
        "What is respiration?",
        "What is the speed of light?"
    ]
    
    print_section("Running Demo Queries")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 80)
        
        try:
            result = pipeline.run(query)
            
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            print(f"⏱️  Timings:")
            for k, v in result["timings"].items():
                print(f"   • {k}: {v:.2f}s")
            
            print(f"\n📚 Sources ({len(result['documents'])} retrieved):")
            for j, doc in enumerate(result["documents"][:3], 1):  # Show top 3
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "unknown")
                cls = doc.metadata.get("class", "unknown")
                print(f"   {j}. {source} (Class {cls}, Page {page})")
            
            if len(result["documents"]) > 3:
                print(f"   ... and {len(result['documents']) - 3} more")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 80)
    
    print("\n✅ Demo completed!")
    print("\n💡 To use the web interface, run:")
    print("   python -m streamlit run app/ui.py")
    print()


if __name__ == "__main__":
    run_demo()