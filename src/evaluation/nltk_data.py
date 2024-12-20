import nltk
import os
from pathlib import Path

def find_writable_dir(paths):
    # Add user's home nltk_data directory to the start of the list
    home_nltk = str(Path.home() / 'nltk_data')
    all_paths = [home_nltk] + paths
    
    for path in all_paths:
        try:
            # Try to create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            # Test if we can write to it
            test_file = os.path.join(path, 'test_write')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                return path
            except (IOError, OSError):
                continue
        except (IOError, OSError):
            continue
    return None

# Get NLTK's default search paths
paths = nltk.data.path

# Find first writable directory
download_dir = find_writable_dir(paths)

if download_dir:
    try:
        nltk.download('punkt', download_dir=download_dir)
        nltk.download('punkt_tab', download_dir=download_dir)
    except Exception as e:
        print(f"Failed to download NLTK data: {str(e)}")
else:
    print("No writable directory found in NLTK's search paths")
