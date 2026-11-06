# TechPulse — AI-Powered Career & Job Market Intelligence Platform

A production-grade career intelligence, machine learning recommendation, and job-market decision platform built with **Streamlit**, **SQLAlchemy 2.x**, **PostgreSQL** (via **Psycopg 3**), **scikit-learn**, **XGBoost**, and **Plotly**, styled with an executive **Red, Black, and Gold** dark SaaS interface.

> **Dataset:** Historical tech job records collected up to **August 2026** (1,722 rows). This is an analytical research dataset, NOT a live vacancy feed.

---

## 🚀 Key Modules & Capabilities

1. **📊 Executive Dashboard (Overview)**
   - Live KPI overview: 1,722 job records, companies, remote distribution, and skills tracked.
   - Domain role distributions and top skills in demand.
   - Quick-access jump cards to all platform tools.

2. **🎯 Career Matcher & Competency Assessment**
   - Multiclass classification powered by an optimized **RandomForest** classifier (`job_role_model_v3.pkl`, **86.38% test accuracy**, **0.8216 Macro F1**).
   - MultiLabelBinarizer feature encoding across 100 standardized technical skills.
   - Transparent, uncalibrated model-estimated class probabilities.
   - Skill Gap breakdown: identifies matched competencies and high-impact missing skills.
   - **Persistent history:** Authenticated users can save analysis records to PostgreSQL with one click.

3. **✨ Intelligent Job Recommendations & Bookmarks**
   - Content-based candidate ranking by skill coverage, TF-IDF text similarity, location, and salary compatibility.
   - Automatically pre-loads technical skills from your saved career profile when logged in.
   - **Saved Jobs:** Bookmark jobs with deduplication (`saved_jobs` table), view enriched details, and open original employer listings.

4. **📄 AI Resume Analyzer**
   - Local in-memory PDF parsing via PyMuPDF (no sensitive resume content is persisted on disk).
   - Heuristic compatibility scoring and gap recommendations against specific job requirements.
   - One-click handoff to Skill Gap Analyzer and save audit results to user account history.

5. **📈 Market Intelligence & Hiring Trends**
   - Interactive Plotly visualizations for top skills, role distributions, remote breakdown, and top hiring companies.
   - Cached data aggregations for zero-latency filter operations.

6. **💰 Salary Benchmark & Prediction**
   - **Salary Benchmark:** Descriptive percentile statistics (P25, P50, P75, P90) across disclosed USD salaries.
   - **Salary Prediction (ML):** Production **XGBoost regressor** (**$23,884 MAE**, **0.5902 R²**) estimating annualized USD compensation with 10–90% error intervals and market comparisons.

7. **💬 AI Career Assistant with Persistent Memory**
   - Grounded career assistant providing structured guidance from real dataset distributions and ML models.
   - **Chat Persistence:** Authenticated sessions are saved to PostgreSQL (`chat_messages` table), allowing users to maintain multi-turn context.
   - Optional external LLM API enhancement via secure environment variables (`TECHPULSE_LLM_API_KEY`).

8. **🔐 Account, Profile Preferences & Community Feedback**
   - Secure authentication with bcrypt password hashing (12 rounds) and email normalization.
   - Persistent user profile preferences (`user_profiles` table) that personalize Career Matcher and Job Recommendations.
   - User analysis history and bookmarked job listings viewer (`my_history.py`).
   - Platform feedback collection with 1–5 star rating validation.

---

## 🗄️ PostgreSQL Database Architecture

TechPulse utilizes a centralized, production-grade PostgreSQL architecture powered by **SQLAlchemy 2.x** and **Psycopg 3**.

### Table Schema Overview

| Table | Description | Key Columns / Constraints |
|---|---|---|
| `users` | User credentials & authentication | `id` (PK), `name`, `email` (UNIQUE), `password_hash`, `created_at` |
| `user_profiles` | Personal career preferences | `id` (PK), `user_id` (FK, UNIQUE), `skills` (TEXT[]), `target_role`, `preferred_location`, `experience_level`, `updated_at` |
| `analysis_history` | Structured assessment history | `id` (PK), `user_id` (FK), `analysis_type`, `result` (JSONB), `created_at` |
| `saved_jobs` | Bookmarked job listings | `id` (PK), `user_id` (FK), `job_id`, `saved_at`, `UNIQUE(user_id, job_id)` |
| `chat_messages` | AI Career Assistant conversations | `id` (PK), `user_id` (FK), `session_id`, `role` (user/assistant), `message`, `created_at` |
| `feedback` | Platform ratings & comments | `id` (PK), `user_id` (FK nullable), `rating` (1-5), `comments`, `created_at` |

### Architectural Highlights
- **Connection Pooling:** SQLAlchemy `QueuePool` with `pool_pre_ping=True`, `pool_size=5`, `max_overflow=10`, `pool_recycle=1800`.
- **URL Normalization:** Automatically adapts `postgres://` or `postgresql://` connection strings into `postgresql+psycopg://`.
- **JSONB Serialization:** Custom `TechPulseJSONEncoder` transparently handles non-native types (NumPy int/float, NumPy arrays, Pandas Timestamps/DataFrames) into native PostgreSQL JSONB.
- **Dual-Mode Engine:** Streamlit resource caching (`@st.cache_resource`) inside the app, with fallback to module singleton for automated tests and scripts.

---

## 🛠️ Project Structure

```text
├── app.py                          # Streamlit entry point with grouped domain navigation
├── app_pages/                      # Core application pages
│   ├── overview.py                 # Executive Red/Black/Gold dashboard
│   ├── career_matcher.py           # ML Job role prediction & profile skill sync
│   ├── job_recommendations.py      # Content-based recommendation engine with bookmarks
│   ├── skill_gap_analyzer.py       # Gap analyzer & learning roadmap
│   ├── resume_analyzer.py          # PDF resume parser & evaluator
│   ├── market_intelligence.py      # Job market distributions & Plotly charts
│   ├── job_explorer.py             # Filterable job directory
│   ├── salary_benchmark.py         # Descriptive compensation percentiles
│   ├── salary_prediction.py        # XGBoost ML salary regressor
│   ├── career_assistant.py         # AI career assistant with persistent chat
│   ├── authentication.py           # Account, profile preferences & feedback
│   ├── my_history.py               # Analysis history & saved jobs viewer
│   └── auth.py                     # Backward-compatibility re-export proxy
├── database/                       # Database schema & migration system
│   ├── schema.sql                  # Complete DDL schema with indexes and constraints
│   ├── migrate.py                  # Idempotent migration runner & schema health inspector
│   └── migrations/                 # Versioned migration scripts
│       └── 001_initial_schema.sql  # Safe, non-destructive initial schema migration
├── utils/
│   ├── db.py                       # Centralized SQLAlchemy 2.x engine & session management
│   ├── auth.py                     # Canonical authentication, profile, history & chat operations
│   ├── theme.py                    # Executive Red, Black & Gold theme engine
│   └── data_loader.py              # Preprocessing, ML loaders & caching
├── src/                            # Machine learning & recommendation pipelines
├── models/                         # Serialized production models (.pkl) & metadata
├── data/                           # Cleaned & processed dataset (1,722 rows)
├── tests/                          # Automated test suite
│   └── test_database.py            # Unit & integration tests (Isolated in-memory test engine)
├── scripts/                        # Model training & test runner scripts
│   └── run_database_tests.py       # Automated database test verification runner
├── .env.example                    # Template environment variables
├── requirements.txt                # Production dependencies
└── README.md                       # Comprehensive documentation
```

---

## ⚡ Setup & Deployment Guides

### 1. Local PostgreSQL Setup

#### Step 1: Install PostgreSQL
Ensure PostgreSQL 14+ is installed and running on your system.
Verify service status:
```powershell
Get-Service -Name postgresql*        # Windows PowerShell
# or: sudo systemctl status postgresql  (Linux)
```

#### Step 2: Create Database & User
Open `psql` (or pgAdmin):
```sql
CREATE USER techpulse_user WITH PASSWORD 'YourSecurePasswordHere';
CREATE DATABASE techpulse_db OWNER techpulse_user;
GRANT ALL PRIVILEGES ON DATABASE techpulse_db TO techpulse_user;
```

#### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env          # Windows
# or: cp .env.example .env           (Linux/macOS)
```
Edit `.env` and set your `DATABASE_URL`:
```text
DATABASE_URL=postgresql+psycopg://techpulse_user:YourSecurePasswordHere@localhost:5432/techpulse_db
```

#### Step 4: Run Migrations
Initialize the schema and verify table health:
```bash
python database/migrate.py
```

#### Step 5: Test Connection
```bash
python test_db.py
```

---

### 2. Cloud Deployment with Neon PostgreSQL

[Neon](https://neon.tech) provides serverless PostgreSQL with instant branching and connection pooling.

#### Step 1: Create a Neon Project
1. Sign in to [console.neon.tech](https://console.neon.tech).
2. Click **Create Project** and name it `techpulse-db`.
3. Choose your preferred cloud region.

#### Step 2: Retrieve Connection String
In the Neon dashboard, copy your pooled connection string:
```text
postgresql://alex:AbC123sK45@ep-sample-pooler.us-east-2.aws.neon.tech/techpulse?sslmode=require
```

#### Step 3: Configure TechPulse for Neon
In your local `.env` file or Streamlit Cloud Secrets:
```text
DATABASE_URL=postgresql+psycopg://alex:AbC123sK45@ep-sample-pooler.us-east-2.aws.neon.tech/techpulse?sslmode=require
```
*(TechPulse automatically normalizes `postgresql://` to `postgresql+psycopg://` if omitted).*

#### Step 4: Run Migrations Against Neon
```bash
python database/migrate.py
```

---

### 3. Streamlit Community Cloud Deployment

#### Step 1: Push Repository to GitHub
Ensure `.env` is NOT committed (it is already included in `.gitignore`).

#### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
2. Select your repository, branch (`main`), and set the main file path to `app.py`.

#### Step 3: Configure Secrets in Streamlit Cloud
Under **App Settings** > **Secrets**, enter your configuration in TOML format:
```toml
DATABASE_URL = "postgresql+psycopg://user:password@ep-sample.us-east-2.aws.neon.tech/techpulse?sslmode=require"

# Optional LLM Key
TECHPULSE_LLM_API_KEY = "your-api-key-here"
```

#### Step 4: Launch
Click **Deploy**. TechPulse will connect to your cloud database automatically.

---

## 🧪 Testing & Verification

### Run the Complete Automated Test Suite
TechPulse includes an isolated automated test suite covering all database operations, encryption, session management, and ML regression tests:

```bash
python scripts/run_database_tests.py
```

Output:
```text
======================================================================
 Verification Results Matrix
======================================================================
  Database connection & pooling                 [PASS]
  User registration & security                  [PASS]
  User profiles & preferences                   [PASS]
  Analysis history & JSONB serialization        [PASS]
  Saved jobs deduplication & lookup             [PASS]
  AI Assistant chat persistence                 [PASS]
  Feedback submission & validation              [PASS]
  Application & ML pipeline integrity           [PASS]
======================================================================
Total Tests Run: 22
Errors: 0 | Failures: 0
======================================================================
```

### Run Standalone CLI Verification Scripts
```bash
python test_db.py             # Test database connectivity
python test_history.py        # Test analysis history saving
python test_profile.py        # Test profile upsert and retrieval
```
