from flask_frozen import Freezer
from app import app  # Assuming your Flask app instance is named 'app'

freezer = Freezer(app)

if __name__ == '__main__':
    freezer.freeze()
