#!/usr/bin/env python
"""
Document Summarizer with Q&A - Async Starter Script
--------------------------------------------------
This script sets up and runs the Document Summarizer with Q&A application with async support.
"""
import os
import sys
import argparse
import logging
import time
import platform
import webbrowser
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("starter")

def check_api_key():
    """Check if OpenRouter API key is set in environment variables."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables or .env file.")
        logger.info("You need to set up your OpenRouter API key to use this application.")
        logger.info("Create a .env file with content: OPENROUTER_API_KEY=your_api_key_here")
        return False
    
    logger.info("✓ OpenRouter API key found.")
    return True

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        "main.py",
        "enhanced_summarizer.py",
        "document_processor.py",
        "openrouter_patch.py",
        "models.py",
        "index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    logger.info("✓ All required files found.")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import fastapi
        import uvicorn
        import aiohttp
        #import aiofiles
        logger.info("✓ Required Python packages found.")
        return True
    except ImportError as e:
        logger.error(f"Missing required Python package: {e.name}")
        logger.info("Please install required packages: pip install fastapi uvicorn aiohttp aiofiles")
        return False

def create_upload_folder():
    """Create upload folder if it doesn't exist."""
    upload_folder = os.environ.get("UPLOAD_FOLDER", "./uploads")
    if not os.path.exists(upload_folder):
        try:
            os.makedirs(upload_folder)
            logger.info(f"✓ Created uploads folder: {upload_folder}")
        except Exception as e:
            logger.error(f"Failed to create uploads folder: {e}")
            return False
    else:
        logger.info(f"✓ Uploads folder exists: {upload_folder}")
    return True

def open_browser(host, port, delay=1.5):
    """Open the browser to the application after a short delay."""
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    
    def _open_browser():
        time.sleep(delay)  # Wait for server to start
        webbrowser.open(url)
        logger.info(f"Opening browser to {url}")
    
    import threading
    threading.Thread(target=_open_browser).start()

def run_application(host, port, open_browser_flag=True, reload=True):
    """Run the FastAPI application with uvicorn."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not found. Please install it with pip install uvicorn")
        return 1
    
    # Create the uploads folder
    if not create_upload_folder():
        return 1
    
    # Print a nice banner
    logger.info("=" * 70)
    logger.info("Document Summarizer with Q&A (Async Version)".center(70))
    logger.info("=" * 70)
    
    # Show access info
    access_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{access_host}:{port}"
    logger.info(f"The application will be available at: {url}")
    
    # Open browser if requested
    if open_browser_flag:
        open_browser(host, port)
    
    # Start uvicorn with the async app
    try:
        uvicorn.run(
            "main:app", 
            host=host, 
            port=port, 
            reload=reload,
            log_level="info"
        )
        return 0
    except Exception as e:
        logger.error(f"Error starting the application: {e}")
        return 1

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start the Document Summarizer with Q&A application (Async Version)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to use")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--check", action="store_true", help="Only check configuration and exit")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check requirements
    api_key_ok = check_api_key()
    files_ok = check_required_files()
    deps_ok = check_dependencies()
    
    # If only checking, exit now
    if args.check:
        if api_key_ok and files_ok and deps_ok:
            logger.info("All checks passed. The application is ready to run.")
            return 0
        else:
            logger.error("Some checks failed. Please fix the issues before running the application.")
            return 1
    
    # Exit if any requirements check failed
    if not (api_key_ok and files_ok and deps_ok):
        logger.error("Cannot start application due to failed requirement checks.")
        return 1
    
    # Run the application
    return run_application(
        host=args.host,
        port=args.port,
        open_browser_flag=not args.no_browser,
        reload=not args.no_reload
    )

if __name__ == "__main__":
    sys.exit(main())