"""
TechPulse Automated Database & Integration Test Runner.

Executes test suites and prints a structured Pass/Fail verification matrix.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_database import (
    TestDatabaseConnection,
    TestUserAuthentication,
    TestUserProfiles,
    TestAnalysisHistory,
    TestSavedJobs,
    TestChatPersistence,
    TestFeedback,
    TestApplicationIntegrity
)
from utils.db import is_database_configured, test_connection


def main():
    print("=" * 70)
    print(" TechPulse: Comprehensive PostgreSQL & Application Test Suite")
    print("=" * 70)

    # 1. Live Database Status
    print("\n[1/3] Checking Live Database Configuration...")
    configured = is_database_configured()
    if configured:
        ok, msg = test_connection()
        if ok:
            print(f"  Live Database Status: [PASS] Online ({msg[:60]}...)")
        else:
            print(f"  Live Database Status: [INFO] Offline ({msg[:80]}...)")
            print("  (Automated tests will run against an isolated test environment)")
    else:
        print("  Live Database Status: [INFO] DATABASE_URL not set in environment/secrets.")
        print("  (Automated tests will run against an isolated test environment)")

    # 2. Run Test Suite
    print("\n[2/3] Running Unit & Integration Test Suite...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDatabaseConnection,
        TestUserAuthentication,
        TestUserProfiles,
        TestAnalysisHistory,
        TestSavedJobs,
        TestChatPersistence,
        TestFeedback,
        TestApplicationIntegrity
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 3. Summary Matrix
    print("\n" + "=" * 70)
    print(" Verification Results Matrix")
    print("=" * 70)
    
    categories = [
        ("Database connection & pooling", TestDatabaseConnection),
        ("User registration & security", TestUserAuthentication),
        ("User profiles & preferences", TestUserProfiles),
        ("Analysis history & JSONB serialization", TestAnalysisHistory),
        ("Saved jobs deduplication & lookup", TestSavedJobs),
        ("AI Assistant chat persistence", TestChatPersistence),
        ("Feedback submission & validation", TestFeedback),
        ("Application & ML pipeline integrity", TestApplicationIntegrity),
    ]

    for label, cls_ref in categories:
        cls_tests = loader.loadTestsFromTestCase(cls_ref)
        cls_names = {t.id().split(".")[-1] for t in cls_tests}
        failed = any(f[0].id().split(".")[-1] in cls_names for f in result.failures + result.errors)
        status = "[FAIL]" if failed else "[PASS]"
        print(f"  {label:<45} {status}")

    print("=" * 70)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Errors: {len(result.errors)} | Failures: {len(result.failures)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

