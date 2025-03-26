import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from mangum import Mangum
from app.main import app

# Configure handler with specific settings for Vercel
handler = Mangum(app, lifespan="off", strip_api_path=True)