import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.auth import save_profile, get_profile
from sqlalchemy import text
from utils.db import get_session

try:
    with get_session() as db:
        user = db.execute(
            text("SELECT id FROM users LIMIT 1")
        ).first()

    if not user:
        print("No registered user found in 'users' table. Register a user first in the app.")
    else:
        user_id = user[0]

        ok, msg = save_profile(
            user_id=user_id,
            skills=["Python", "SQL", "Pandas", "Machine Learning"],
            target_role="Data Scientist",
            location="Delhi",
            experience="Fresher"
        )

        if ok:
            profile = get_profile(user_id)
            print("Saved profile successfully:")
            print(profile)
        else:
            print("Failed to save profile:", msg)

except Exception as e:
    print("Database connection failed:")
    print(e)
