import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.db import test_connection

success, message = test_connection()
if success:
    print("SUCCESS: PostgreSQL connected successfully!")
    print("Database version:", message)
else:
    print("FAILURE: Database connection failed.")
    print("Reason:", message)
