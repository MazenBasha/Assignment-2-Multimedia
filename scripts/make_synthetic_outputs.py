"""Fabricate synthetic Mode A + Mode B output JSONL files so we can smoke-test
the eval pipeline on a machine without GPU / data / API keys.

NOT for real evaluation — the reports here are made up. Use to verify that
metrics_report.py, metrics_vqa.py, and qualitative_dump.py wire correctly.
"""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ---------------------------- Mode A : reports ----------------------------

MODE_A_CASES = [
    {
        "study_id": "s001",
        "image_path": "data/mimic-cxr/images/s001__a.jpg",
        "reference": (
            "FINDINGS: The lungs are clear without focal consolidation. "
            "No pleural effusion or pneumothorax. The cardiomediastinal "
            "silhouette is normal in size.\n"
            "IMPRESSION: 1. No acute cardiopulmonary process."
        ),
        # RAG: stays close to reference (retrieval helps surface common phrases)
        "rag_pred": (
            "FINDINGS: The lungs are clear with no focal consolidation. "
            "There is no pleural effusion or pneumothorax. The cardiomediastinal "
            "silhouette is normal.\n"
            "IMPRESSION: 1. No acute cardiopulmonary process."
        ),
        # Baseline: more generic, vaguer
        "base_pred": (
            "FINDINGS: Lungs appear clear. No obvious effusion. Heart size "
            "appears within normal limits.\n"
            "IMPRESSION: Unremarkable chest radiograph."
        ),
    },
    {
        "study_id": "s002",
        "image_path": "data/mimic-cxr/images/s002__a.jpg",
        "reference": (
            "FINDINGS: There is a small right pleural effusion with adjacent "
            "atelectasis at the right lung base. The left lung is clear. The "
            "heart size is mildly enlarged.\n"
            "IMPRESSION: 1. Small right pleural effusion. 2. Mild cardiomegaly."
        ),
        "rag_pred": (
            "FINDINGS: Small right pleural effusion with basilar atelectasis. "
            "The left lung is clear. Heart is mildly enlarged.\n"
            "IMPRESSION: 1. Small right pleural effusion. "
            "2. Mild cardiomegaly."
        ),
        "base_pred": (
            "FINDINGS: There may be a haziness over the right lower lung. "
            "The cardiac silhouette is borderline.\n"
            "IMPRESSION: Possible right basal opacity."
        ),
    },
    {
        "study_id": "s003",
        "image_path": "data/mimic-cxr/images/s003__a.jpg",
        "reference": (
            "FINDINGS: A right-sided central venous catheter terminates in "
            "the superior vena cava. No pneumothorax. The lungs are clear.\n"
            "IMPRESSION: 1. CVC in good position. 2. No acute findings."
        ),
        # RAG hallucinates a device the baseline correctly skips — RAG-loss case.
        "rag_pred": (
            "FINDINGS: A right-sided central venous catheter terminates in "
            "the right atrium. A left-sided chest tube is in place. Bilateral "
            "lower lobe atelectasis is present.\n"
            "IMPRESSION: 1. Malpositioned CVC. 2. Bilateral atelectasis."
        ),
        "base_pred": (
            "FINDINGS: A right-sided central venous catheter terminates in "
            "the superior vena cava. No pneumothorax. Lungs clear.\n"
            "IMPRESSION: CVC in good position. No acute findings."
        ),
    },
    {
        "study_id": "s004",
        "image_path": "data/mimic-cxr/images/s004__a.jpg",
        "reference": (
            "FINDINGS: New patchy opacity in the right lower lobe concerning "
            "for pneumonia. No effusion or pneumothorax. Heart size normal.\n"
            "IMPRESSION: 1. Right lower lobe pneumonia."
        ),
        "rag_pred": (
            "FINDINGS: Patchy opacity in the right lower lobe concerning for "
            "pneumonia. No pleural effusion. Heart size is normal.\n"
            "IMPRESSION: 1. Right lower lobe pneumonia."
        ),
        "base_pred": (
            "FINDINGS: Asymmetry in the lung bases. Heart appears normal.\n"
            "IMPRESSION: Findings of unclear etiology."
        ),
    },
    {
        "study_id": "s005",
        "image_path": "data/mimic-cxr/images/s005__a.jpg",
        "reference": (
            "FINDINGS: Mild pulmonary edema with cephalization of pulmonary "
            "vasculature. Bilateral small pleural effusions. The cardiac "
            "silhouette is enlarged.\n"
            "IMPRESSION: 1. Pulmonary edema. 2. Bilateral pleural effusions. "
            "3. Cardiomegaly."
        ),
        "rag_pred": (
            "FINDINGS: Mild pulmonary edema with cephalization. Bilateral "
            "small pleural effusions. Heart is enlarged.\n"
            "IMPRESSION: 1. Pulmonary edema. 2. Small bilateral effusions. "
            "3. Cardiomegaly."
        ),
        "base_pred": (
            "FINDINGS: Increased lung markings bilaterally. Heart appears "
            "enlarged.\n"
            "IMPRESSION: Findings could represent congestive failure."
        ),
    },
    {
        "study_id": "s006",
        "image_path": "data/mimic-cxr/images/s006__a.jpg",
        "reference": (
            "FINDINGS: Large left-sided pneumothorax with associated mediastinal "
            "shift to the right. No focal consolidation. Heart size normal.\n"
            "IMPRESSION: 1. Large left tension pneumothorax requiring urgent "
            "intervention."
        ),
        # RAG retrieval helps because pneumothorax phrasing is consistent.
        "rag_pred": (
            "FINDINGS: Large left-sided pneumothorax with rightward mediastinal "
            "shift. No focal consolidation. Heart size is normal.\n"
            "IMPRESSION: 1. Large left tension pneumothorax."
        ),
        "base_pred": (
            "FINDINGS: Asymmetric lucency over the left hemithorax. Possible "
            "displacement of mediastinal structures.\n"
            "IMPRESSION: Findings concerning for pneumothorax, recommend "
            "clinical correlation."
        ),
    },
]


def write_mode_a() -> None:
    rag_rows = []
    base_rows = []
    for c in MODE_A_CASES:
        common = {
            "study_id":   c["study_id"],
            "subject_id": "p" + c["study_id"][1:],
            "image_path": c["image_path"],
            "reference":  c["reference"],
        }
        rag_rows.append({
            **common,
            "prediction": c["rag_pred"],
            "system":     "rag",
            "retrieved": [
                {"rank": 0, "score": 8.2, "study_id": f"r{c['study_id'][1:]}a",
                 "subject_id": "pX", "image_path": "data/.../X.jpg", "text": "FINDINGS: ... IMPRESSION: ..."},
                {"rank": 1, "score": 7.9, "study_id": f"r{c['study_id'][1:]}b",
                 "subject_id": "pY", "image_path": "data/.../Y.jpg", "text": "FINDINGS: ... IMPRESSION: ..."},
            ],
        })
        base_rows.append({
            **common,
            "prediction": c["base_pred"],
            "system":     "baseline",
            "retrieved":  [],
        })

    (OUT / "mode_a_rag.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rag_rows) + "\n", encoding="utf-8")
    (OUT / "mode_a_baseline.jsonl").write_text(
        "\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    print(f"Mode A: wrote {len(rag_rows)} RAG + {len(base_rows)} baseline rows.")


# ---------------------------- Mode B : VQA --------------------------------

MODE_B_CASES = [
    # presence
    ("q1", "s002", "presence",     "Is there a pleural effusion?",
     "Yes, small right pleural effusion.",
     "Yes, small right pleural effusion.",
     "Yes, possibly some haziness at the right base."),
    ("q2", "s001", "presence",     "Is there a pneumothorax?",
     "No.",
     "No.",
     "No."),
    ("q3", "s006", "presence",     "Is there a pneumothorax?",
     "Yes, large left pneumothorax.",
     "Yes, large left pneumothorax.",
     "Possibly, left hemithorax asymmetry."),
    # location
    ("q4", "s004", "location",     "Where is the consolidation?",
     "Right lower lobe.",
     "Right lower lobe.",
     "Lower lung field."),
    ("q5", "s002", "location",     "Which side has the effusion?",
     "Right.",
     "Right.",
     "Cannot be determined from the image."),
    # severity
    ("q6", "s005", "severity",     "How severe is the pulmonary edema?",
     "Mild.",
     "Mild.",
     "Moderate."),
    ("q7", "s002", "severity",     "How severe is the cardiomegaly?",
     "Mild.",
     "Mild.",
     "Borderline."),
    # differential
    ("q8", "s004", "differential", "What could explain the right basal opacity?",
     "Pneumonia.",
     "Pneumonia or aspiration.",
     "Effusion or atelectasis."),
    # description
    ("q9", "s003", "description",  "Describe the position of the central venous catheter.",
     "Terminates in the superior vena cava.",
     "Terminates in the right atrium.",
     "Terminates in the superior vena cava."),
    ("q10","s001", "description",  "Describe the cardiac silhouette.",
     "Normal in size.",
     "Normal in size.",
     "Within normal limits."),
]


def write_mode_b() -> None:
    rag_rows, base_rows = [], []
    for qid, sid, qtype, q, gold, rag_pred, base_pred in MODE_B_CASES:
        common = {
            "question_id":   qid,
            "study_id":      sid,
            "subject_id":    "p" + sid[1:],
            "image_path":    f"data/mimic-cxr/images/{sid}__a.jpg",
            "question":      q,
            "question_type": qtype,
            "reference":     gold,
        }
        rag_rows.append({**common, "prediction": rag_pred, "system": "rag", "retrieved": []})
        base_rows.append({**common, "prediction": base_pred, "system": "baseline", "retrieved": []})

    (OUT / "mode_b_rag.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rag_rows) + "\n", encoding="utf-8")
    (OUT / "mode_b_baseline.jsonl").write_text(
        "\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8")
    print(f"Mode B: wrote {len(rag_rows)} RAG + {len(base_rows)} baseline rows.")


if __name__ == "__main__":
    write_mode_a()
    write_mode_b()
