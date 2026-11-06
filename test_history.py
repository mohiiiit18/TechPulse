import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.auth import add_history

# Replace with your actual user ID in PostgreSQL
user_id = 1

test_result = {
    "skills": ["Python", "SQL", "Machine Learning"],
    "predicted_role": "Data Scientist",
    "top_score": 85.5
}

ok, msg = add_history(
    user_id,
    "career_match",
    test_result
)

if ok:
    print("SUCCESS: Career prediction saved to database!")
else:
    print("Database insert failed:")
    print(msg)
