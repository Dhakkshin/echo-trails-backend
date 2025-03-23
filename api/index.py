from mangum import Mangum
from app.main import app

# Create handler for AWS Lambda & API Gateway
handler = Mangum(app)