"""Skill normalization + de-duplication (shared by display and search)."""

import re


def normalize_skill(skill: str) -> str:
    """'Machine-Learning' / 'machine_learning ' -> 'machine learning'."""
    s = (skill or "").lower().strip()
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^\w\s+#.]", "", s)  # keep c++, c#, .net, node.js
    return re.sub(r"\s+", " ", s).strip()


def merge_skills(*skill_lists) -> list[str]:
    """Combine several skill lists, dropping duplicates (first spelling wins)."""
    seen: dict[str, str] = {}
    for skills in skill_lists:
        for skill in skills or []:
            key = normalize_skill(skill)
            if key and key not in seen:
                seen[key] = skill.strip()
    return list(seen.values())

def compact_skill(skill: str) -> str:
    """'Py Spark' / 'PySpark' / 'py-spark' -> 'pyspark'; 'Azure ML' -> 'azureml'."""
    return normalize_skill(skill).replace(" ", "")

