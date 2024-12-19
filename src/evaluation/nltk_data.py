import nltk
import os

# Get venv path and create nltk_data directory
venv_dir = os.path.join(os.environ['VIRTUAL_ENV'], 'nltk_data')
os.makedirs(venv_dir, exist_ok=True)

# Add path for NLTK to find data
nltk.data.path.append(venv_dir)

# Download required data to venv directory
nltk.download('punkt', download_dir=venv_dir)
nltk.download('punkt_tab', download_dir=venv_dir)

