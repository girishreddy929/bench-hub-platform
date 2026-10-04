"""Checks for the parts of ai_engine that need no model. Run: python -m tests.test_ai_engine"""

from src.ai import ai_engine as ai

# ---- text cleaning: bullets, blank lines, repeated spaces ----
assert ai.clean_text("  •  Python    SQL  \n\n ● \n- Spark") == "Python SQL\nSpark"

# ---- certifications: Girish's resume layout ----
GIRISH = """Girish Reddy Sura
Data Scientist
Skills
• Python
• PySpark
Experience
Associate Manager
Tredence Analytics, Bangalore, India
Key Projects
• Rebate.AI – AI Powered Product Categorization Platform
Certifications
- Iguazio MLOps Certification – Iguazio
- Generative AI Associate – Databricks
- LLM Engineering (Industry Program)
- Demand Forecasting (Industry Program)
Hobbies And Interests
• Graphics & AR
"""
certs = ai.extract_certifications(GIRISH, "")
assert [c["name"] for c in certs] == [
    "Iguazio MLOps Certification",
    "Generative AI Associate",
    "LLM Engineering (Industry Program)",
    "Demand Forecasting (Industry Program)",
], certs
assert certs[0]["issuer"] == "Iguazio" and certs[1]["issuer"] == "Databricks" and certs[2]["issuer"] == ""

# ---- certifications on the heading line, with a year ----
inline = ai.extract_certifications("Certifications: AWS Solutions Architect (2023); Azure DP-100", "")
assert [c["name"] for c in inline] == ["AWS Solutions Architect", "Azure DP-100"], inline
assert inline[0]["year"] == "2023"

# ---- name, issuer and year in one line ----
full = ai.extract_certifications(
    "Certifications\nGoogle Professional Data Engineer - Google Cloud - 2022", ""
)
assert full == [{"name": "Google Professional Data Engineer", "issuer": "Google Cloud", "year": "2022"}], full

# ---- a certification that starts with a heading word is not a heading ----
pmp = ai.extract_certifications(
    "Certifications\nProject Management Professional (PMP) – PMI\nEducation\nB.Tech", ""
)
assert [c["name"] for c in pmp] == ["Project Management Professional (PMP)"], pmp
assert pmp[0]["issuer"] == "PMI"

# ---- no heading anywhere: short 'certified' lines only ----
fallback = ai.extract_certifications("", "Built models.\nAWS Certified Solutions Architect\n")
assert [c["name"] for c in fallback] == ["AWS Certified Solutions Architect"], fallback

# ---- a sentence is not a certification ----
prose = (
    "Certification in Azure ML was successfully completed during the internal "
    "training program last year."
)
assert ai.extract_certifications(prose, "") == []
assert ai.extract_certifications("", "") == []

# ---- input hash: formatting noise ignored, real changes detected ----
h1 = ai.input_hash("resume text", "slider text", "m1")
assert h1 == ai.input_hash("resume    text", "slider text", "m1")
assert h1 != ai.input_hash("resume text 2", "slider text", "m1")
assert h1 != ai.input_hash("resume text", "slider text", "m2")

# ---- both calls share the same document prefix, so the model can reuse it ----
p1 = ai.build_prompt(ai.SUMMARY_TASK, "R", "S")
p2 = ai.build_prompt(ai.PROJECTS_TASK, "R", "S")
shared = p1.split("### TASK")[0]
assert p2.startswith(shared) and "### TASK" in p2

# ---- project cleaning: cap of 4, duplicates merged, bad source fixed, junk dropped ----
raw = (
    [
        {"name": "A", "role": "", "description": "d", "technologies": ["Python", "python"], "source": "weird"},
        {"name": "", "description": ""},
        "junk",
    ]
    + [{"name": f"P{i}", "description": "x"} for i in range(6)]
)
projects = ai._clean_projects(raw)
assert len(projects) == 4
assert projects[0]["technologies"] == ["Python"] and projects[0]["source"] == "resume"

print("All AI-output checks passed.")