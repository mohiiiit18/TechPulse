"""TechPulse Phase 5 — Reusable technical skill dictionary & extractor.

Category-grouped skill vocabulary with alias normalization (e.g.
"ml" -> "machine-learning", "js" -> "javascript"). Matching is done on
normalized tokens; longest-alias-first matching against resume text so
"machine learning" beats its component words. Nothing is invented: a skill is
extracted only if a known alias literally appears in the text.
"""

from __future__ import annotations

import re
from collections import OrderedDict

# Category -> canonical skills (with alias lists for normalization).
# Aliases map to the canonical tokens used across TechPulse (dataset `parsed_skills`
# vocabulary where possible, so extractor output plugs directly into the recommender).
SKILL_DICTIONARY: dict[str, dict[str, list[str]]] = {
    "Programming Languages": {
        "python": ["python", "python3"],
        "javascript": ["javascript", "js", "node.js", "nodejs"],
        "typescript": ["typescript", "ts"],
        "java": ["java", "core java"],
        "c#": ["c#", "csharp", ".net", "dotnet"],
        "c++": ["c++", "cpp"],
        "go": ["go", "golang"],
        "rust": ["rust"],
        "ruby": ["ruby", "ruby on rails"],
        "php": ["php", "laravel"],
        "swift": ["swift", "swiftui"],
        "kotlin": ["kotlin"],
        "scala": ["scala"],
        "sql": ["sql", "t-sql", "pl/sql"],
        "r": ["r language", " r ", "rstudio"],
        "html-css": ["html", "css", "html/css", "html5", "css3", "sass", "scss"],
    },
    "Data Science": {
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "statistics": ["statistics", "statistical analysis", "statistical modeling"],
        "data-science": ["data science", "data analytics", "data analysis"],
        "data-engineering": ["data engineering", "data pipelines", "etl"],
        "etl": ["etl", "elt", "data pipeline"],
        "excel": ["excel", "spreadsheet"],
        "gis": ["gis", "geospatial"],
        "datawrangling": ["data wrangling", "data cleaning"],
    },
    "Machine Learning": {
        "machine-learning": ["machine learning", "ml", "predictive modeling"],
        "deep-learning": ["deep learning", "neural networks"],
        "nlp": ["nlp", "natural language processing", "text mining"],
        "computer-vision": ["computer vision", "opencv", "image processing"],
        "tensorflow": ["tensorflow", "keras"],
        "pytorch": ["pytorch", "torch"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "llm": ["llm", "large language model", "gpt", "prompt engineering", "genai", "generative ai"],
        "reinforcement-learning": ["reinforcement learning", "rlhf"],
        "mlops": ["mlops", "model deployment", "model serving"],
    },
    "Databases": {
        "postgresql": ["postgresql", "postgres"],
        "mysql": ["mysql", "mariadb"],
        "mongodb": ["mongodb", "mongo"],
        "redis": ["redis"],
        "elasticsearch": ["elasticsearch", "opensearch"],
        "snowflake": ["snowflake"],
        "bigquery": ["bigquery"],
        "redshift": ["redshift"],
        "databricks": ["databricks"],
        "nosql": ["nosql", "dynamodb", "cassandra"],
        "sqlserver": ["sql server", "mssql"],
        "sqlite": ["sqlite"],
    },
    "Visualization": {
        "tableau": ["tableau"],
        "power-bi": ["power bi", "powerbi"],
        "looker": ["looker"],
        "matplotlib": ["matplotlib", "seaborn", "ggplot"],
        "dashboards": ["dashboards", "dashboarding", "data visualization", "dataviz"],
    },
    "Cloud Computing": {
        "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
        "azure": ["azure", "microsoft azure"],
        "gcp": ["gcp", "google cloud"],
        "docker": ["docker", "containerization", "containers"],
        "kubernetes": ["kubernetes", "k8s"],
        "serverless": ["serverless", "faas"],
        "terraform": ["terraform", "infrastructure as code", "iac"],
        "ci-cd": ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci"],
    },
    "Development Tools": {
        "git": ["git", "github", "gitlab", "bitbucket", "version control"],
        "linux": ["linux", "unix", "bash", "shell scripting"],
        "rest-api": ["rest api", "restful", "api development", "graphql", "grpc"],
        "microservices": ["microservices", "microservices architecture"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "spring": ["spring", "spring boot"],
        "react": ["react", "react.js", "reactjs", "next.js", "nextjs"],
        "vue": ["vue", "vue.js", "nuxt"],
        "angular": ["angular", "angularjs"],
        "agile": ["agile", "scrum", "kanban", "jira"],
        "testing": ["unit testing", "pytest", "jest", "selenium", "test automation", "qa"],
        "airflow": ["airflow", "airflow orchestration"],
        "spark": ["spark", "pyspark", "hadoop"],
        "kafka": ["kafka", "event streaming"],
    },
}

# Certifications vocabulary (literal substring detection).
CERTIFICATION_KEYWORDS = [
    "aws certified", "azure certified", "google cloud certified", "professional cloud",
    "pmp", "certified scrum master", "csm", "safe", "prince2",
    "comptia", "ccna", "ccnp", "security+",
    "cfa", "frm", "cpa",
    "tensorflow developer certificate", "azure data scientist", "databricks certified",
    "salesforce certified", "servicenow certified", "cisco certified",
]

_SECTION_HEADERS = {
    "education": ["education", "academic background", "academics", "qualifications",
                  "educational qualification"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "career history", "work history"],
    "skills": ["skills", "technical skills", "skill set", "core competencies",
               "technologies", "technical expertise"],
    "projects": ["projects", "personal projects", "key projects", "project experience",
                 "selected projects"],
    "certifications": ["certifications", "certificates", "licenses",
                       "professional development", "courses & certifications"],
}

_WORD_RE = re.compile(r"[a-z0-9+#.\-/ ]+")


def normalize_skill(token: str) -> str | None:
    """Map any alias to its canonical skill token. Returns None if unknown."""
    t = " " + re.sub(r"\s+", " ", str(token).strip().lower()) + " "
    for category, skills in SKILL_DICTIONARY.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                a = alias if alias.startswith(" ") or alias.endswith(" ") else f" {alias} "
                if a in t or t.strip() == alias:
                    return canonical
    return None


def _flatten_aliases() -> list[tuple[str, str, str]]:
    """(alias_for_regex, canonical, category) sorted longest-first."""
    flat = []
    for category, skills in SKILL_DICTIONARY.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                flat.append((alias.strip(), canonical, category))
    flat.sort(key=lambda x: -len(x[0]))
    return flat


_FLAT = _flatten_aliases()


def extract_skills(text: str) -> dict[str, list[str]]:
    """Extract known technical skills from free text, grouped by category.

    Only literal alias occurrences count — nothing inferred or invented.
    Duplicates collapse to one canonical entry per category.
    """
    if not text or not str(text).strip():
        return {}
    lowered = str(text).lower()
    # Guard against pathological matches inside long words (e.g. "r").
    padded = re.sub(r"([a-z0-9+#.\-/]+)", lambda m: f" {m.group(1)} ", lowered)

    found: dict[str, list[str]] = OrderedDict()
    for alias, canonical, category in _FLAT:
        if canonical in found.get(category, []):
            continue
        pattern = r"(?<![a-z0-9+#])" + re.escape(alias).replace(r"\ ", r"\s+") + r"(?![a-z0-9+#])"
        if re.search(pattern, padded if len(alias) <= 3 else lowered):
            found.setdefault(category, []).append(canonical)
    return found


def extract_certifications(text: str) -> list[str]:
    """Return certification keywords literally present in the text."""
    if not text:
        return []
    lowered = str(text).lower()
    return [kw for kw in CERTIFICATION_KEYWORDS if kw in lowered]


def find_sections(text: str) -> dict[str, str]:
    """Locate common resume sections by header keywords.

    Returns {section_name: section_text} for headers found. Content between a
    header and the next recognized header (or end) is that section's text.
    Unrecognized layouts simply yield fewer sections (never fabricated).
    """
    if not text:
        return {}
    lines = str(text).splitlines()
    # Map each line index to a section if the line IS (mostly) a header.
    header_at: dict[int, str] = {}
    for i, line in enumerate(lines):
        clean = re.sub(r"[^a-z ]", "", line.strip().lower()).strip()
        if not clean or len(clean) > 40:
            continue
        for section, headers in _SECTION_HEADERS.items():
            if any(clean == h or (len(clean) > 3 and clean.startswith(h) and len(clean) < len(h) + 12)
                   for h in headers):
                header_at[i] = section
                break

    sections: dict[str, str] = {}
    order = sorted(header_at.items())
    for j, (start_line, section) in enumerate(order):
        end_line = order[j + 1][0] if j + 1 < len(order) else len(lines)
        body = "\n".join(lines[start_line + 1:end_line]).strip()
        # Keep the longest body when a section header appears twice.
        if section not in sections or len(body) > len(sections[section]):
            sections[section] = body
    return sections
