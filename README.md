# TechPulse — AI-Powered Career & Job Market Intelligence Platform

A production-grade career intelligence and job-market decision platform built with **Streamlit**, **scikit-learn**, **XGBoost**, and **Plotly**, styled with an executive **Red, Black, and Gold** dark SaaS interface.

> **Dataset:** Historical tech job records collected up to **August 2026** (1,722 rows). This is an analytical research dataset, NOT a live vacancy feed.

---

## 🚀 Key Modules & Capabilities

1. **📊 Executive Dashboard (Overview)**
   - Live KPI overview: 1,722 job records, companies, remote distribution, and skills tracked.
   - Domain role distributions and top skills in demand.
   - Quick-access jump cards to all platform tools.

2. **🎯 Career Matcher & Skill Gap Analysis**
   - Multiclass classification powered by an optimized **RandomForest** classifier (`job_role_model_v3.pkl`, **86.38% test accuracy**, **0.8216 Macro F1**).
   - MultiLabelBinarizer feature encoding across 100 standardized technical skills.
   - Transparent, uncalibrated model-estimated class probabilities.
   - Skill Gap breakdown: identifies matched competencies and high-impact missing skills.

3. **✨ Intelligent Job Recommendations**
   - Content-based candidate ranking by skill coverage, TF-IDF text similarity, location, and salary compatibility.
   - Directly links to verified employer postings.

4. **📄 AI Resume Analyzer**
   - Local in-memory PDF parsing via PyMuPDF (no sensitive resume content is persisted on disk).
   - Heuristic compatibility scoring and gap recommendations against specific job requirements.

5. **📈 Market Intelligence & Hiring Trends**
   - Interactive Plotly visualizations for top skills, role distributions, remote breakdown, and top hiring companies.
   - Cached data aggregations for zero-latency filter operations.

6. **💰 Salary Benchmark & Prediction**
   - **Salary Benchmark:** Descriptive percentile statistics (P25, P50, P75, P90) across disclosed USD salaries.
   - **Salary Prediction (ML):** Production **XGBoost regressor** (**$23,884 MAE**, **0.5902 R²**) estimating annualized USD compensation with 10–90% error intervals and market comparisons.

7. **💬 AI Career Assistant**
   - Grounded career assistant providing structured guidance from real dataset distributions and ML models.
   - Optional external LLM API enhancement via secure environment variables (`TECHPULSE_LLM_API_KEY`).

---

## 🛠️ Project Structure

```text
├── app.py                          # Streamlit entry point with domain navigation
├── app_pages/                      # 10 core application pages
│   ├── overview.py                 # Executive Red/Black/Gold dashboard
│   ├── career_matcher.py           # ML Job role prediction
│   ├── job_recommendations.py      # Content-based recommendation engine
│   ├── skill_gap_analyzer.py       # Gap analyzer & learning roadmap
│   ├── resume_analyzer.py          # PDF resume parser & evaluator
│   ├── market_intelligence.py      # Job market distributions & Plotly charts
│   ├── job_explorer.py             # Filterable job directory
│   ├── salary_benchmark.py         # Descriptive compensation percentiles
│   ├── salary_prediction.py        # XGBoost ML salary regressor
│   └── career_assistant.py         # Offline/online career chatbot
├── utils/
│   ├── theme.py                    # Centralized Red, Black & Gold theme engine
│   └── data_loader.py              # Preprocessing, ML loaders & caching
├── src/                            # Machine learning & recommendation pipelines
├── models/                         # Serialized production models (.pkl) & metadata
├── data/                           # Cleaned & processed dataset (1,722 rows)
├── scripts/                        # Model training & test suites
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Comprehensive documentation
```

---

## ⚡ Quickstart

### 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# or: source .venv/bin/activate  (Linux/macOS)
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Testing & Quality Assurance
Run the comprehensive automated test suite (Unit, Integration, Regression, and AppTest):
```bash
python scripts/run_tests.py
python scripts/test_recommendations.py
python scripts/test_skill_gap.py
python scripts/test_salary_prediction.py
python scripts/test_phase10_ui.py
```

---

## 🎨 Theme: Red, Black, and Gold
The entire platform is styled using a modern dark SaaS visual hierarchy:
- **Background:** `#080808` (Pitch Black)
- **Secondary Background:** `#111111` (Jet Black)
- **Card Background:** `#151515` (Elevated Card Black)
- **Primary Red:** `#E32636`
- **Deep Red:** `#8B0000`
- **Gold:** `#D4AF37`
- **Light Gold:** `#FFD76A`
- **Primary Text:** `#FFFFFF`
- **Secondary Text:** `#B8B8B8`
- **Borders:** `#3A2020`
