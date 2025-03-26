import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from mangum import Mangum
from app.main import app

# Create handler for Vercel
handler = Mangum(app, lifespan="off")