# Mode A — qualitative examples

## RAG wins

### study_id=1371  Δtoken_F1=+0.325

**Image**: `data\openi\images\1371__CXR1371_IM-0239-1001.png`

**Reference**:

```
IMPRESSION: Heart size is normal. Mildly tortuous aorta, otherwise mediastinal silhouette and pulmonary vascularity are within normal limits. No focal airspace consolidation of pneumonia, pleural effusion or pneumothorax. Couple of XXXX nodules in the left base are XXXX calcified sequela of old granulomatous disease. No pleural effusion or pneumothorax.
```

**RAG prediction**:

```
INDICATION: XXXX

FINDINGS: Lungs are clear bilaterally. There is no focal consolidation, pleural effusion, or pneumothoraces. Cardiomediastinal silhouette is within normal limits. XXXX are unremarkable.

IMPRESSION: No acute cardiopulmonary abnormality.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 3059, 2471, 1467

---

### study_id=1124  Δtoken_F1=+0.324

**Image**: `data\openi\images\1124__CXR1124_IM-0081-2001.png`

**Reference**:

```
FINDINGS: There are postoperative changes of sternotomy. Heart size is within normal limits. There is aortic atherosclerotic vascular calcification. The lungs are mildly hyperexpanded. There is no focal airspace consolidation. No pleural effusion or pneumothorax. There are diffuse degenerative changes of the spine.
IMPRESSION: 1. No focal airspace consolidation. 2. Hyperexpanded lungs, suggestive of obstructive lung disease.
```

**RAG prediction**:

```
INDICATION: Preoperative evaluation

FINDINGS: Cardiomegaly with unfolded aorta. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a large pleural effusion. There is no evidence of pneumothorax.

IMPRESSION: Cardiomegaly. Clear lungs. .
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 686, 3966, 2471

---

### study_id=1014  Δtoken_F1=+0.285

**Image**: `data\openi\images\1014__CXR1014_IM-0013-2001.png`

**Reference**:

```
IMPRESSION: Slightly enlarged heart. No effusions. No edema. No nodules or masses. Aortic XXXX calcification. Aortic XXXX is normal size. Lungs are clear.
```

**RAG prediction**:

```
INDICATION: XXXX XXXX UP;

FINDINGS: The heart is normal in size. The mediastinum is unremarkable. The lungs are clear.

IMPRESSION: No acute disease.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 1461, 583, 81

---

### study_id=2030  Δtoken_F1=+0.270

**Image**: `data\openi\images\2030__CXR2030_IM-0675-2001.png`

**Reference**:

```
FINDINGS: There is hyperinflation lungs due to small calcification is seen posteriorly in the right which may be pleural. There is no pleural effusion or pneumothorax. The heart is not significantly enlarged. There are atherosclerotic changes of the aorta. Arthritic changes of the skeletal structures are noted.
IMPRESSION: COPD. No acute pulmonary disease.
```

**RAG prediction**:

```
INDICATION: XXXX XXXX UP;

FINDINGS: The heart is normal in size. The mediastinum is unremarkable. The lungs are clear.

IMPRESSION: No acute disease.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 1461, 1987, 296

---

### study_id=2443  Δtoken_F1=+0.250

**Image**: `data\openi\images\2443__CXR2443_IM-0980-1001.png`

**Reference**:

```
FINDINGS: Heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen. Vascular calcification is noted.
IMPRESSION: 1. No evidence of active disease.
```

**RAG prediction**:

```
INDICATION: Injury, XXXX XXXX XXXX on patient, bone pain, anemia

FINDINGS: The heart is normal in size. The mediastinum is stable with aortic tortuosity. Lungs are clear with no pleural effusion or pneumothorax. No displaced rib fractures are noted. There are multilevel degenerative changes of the thoracic spine.

IMPRESSION: No acute disease.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 3019, 73, 1218

---

## RAG losses

### study_id=3235  Δtoken_F1=-0.147

**Image**: `data\openi\images\3235__CXR3235_IM-1532-1001.png`

**Reference**:

```
FINDINGS: Heart size is normal. The lungs are clear. No pneumothorax or pleural effusion.
IMPRESSION: Normal chest exam.
```

**RAG prediction**:

```
INDICATION: XXXX-year-old female, right heart failure.

FINDINGS: Cardiomegaly. No focal consolidation. No pleural effusions. No evidence of pneumothorax. Osseous structures intact. Levocurvature of the thoracic spine. Lumbar vertebral body stabilization XXXX.

IMPRESSION: 1. No acute cardiopulmonary abnormality. 2. Cardiomegaly. .
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 3941, 567, 3140

---

### study_id=1450  Δtoken_F1=-0.087

**Image**: `data\openi\images\1450__CXR1450_IM-0291-2001.png`

**Reference**:

```
FINDINGS: The cardiomediastinal silhouette is normal in size and contour. No focal consolidation, pneumothorax or large pleural effusion.
IMPRESSION: Negative for acute abnormality.
```

**RAG prediction**:

```
INDICATION: XXXX-year-old male with chest pain.

FINDINGS: The heart and mediastinum are unremarkable. Again identified are numerous calcified mediastinal lymph XXXX as well as large calcifications within the left upper and left lower lobes. These appear similar to the patient's previous chest CT and are XXXX the sequela of prior granulomatous disease. The lungs are otherwise clear without infiltrate. There is no effusion or pneumothorax.

IMPRESSION: 1. Evidence of prior granulomatous disease, stable. No acute abnormality.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 1816, 1099, 2354

---

### study_id=223  Δtoken_F1=-0.016

**Image**: `data\openi\images\223__CXR223_IM-0831-2001.png`

**Reference**:

```
FINDINGS: Heart size within normal limits, stable mediastinal and hilar contours. No focal alveolar consolidation, no definite pleural effusion seen. Bronchovascular crowding without typical findings of pulmonary edema.
IMPRESSION: No acute findings
```

**RAG prediction**:

```
INDICATION: History of XXXX implant

IMPRESSION: Heart size is normal. Minimal subsegmental atelectasis in the left base, seen XXXX on the lateral film, otherwise lungs are clear. Degenerative disease of thoracic spine. Stable left lower lobe 5 mm granuloma adjacent to the heart XXXX.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 3799, 3098, 1897

---

### study_id=1596  Δtoken_F1=+0.013

**Image**: `data\openi\images\1596__CXR1596_IM-0387-1001.png`

**Reference**:

```
FINDINGS: Normal heart. Clear lungs. Stable calcified granuloma left midlung. No pneumothorax. No pleural effusion. Midline trachea.
IMPRESSION: Normal chest exam.
```

**RAG prediction**:

```
INDICATION: chest pain

FINDINGS: The trachea is midline. The cardiomediastinal silhouette is normal and unchanged compared to prior examination. Lungs are clear, without evidence of acute infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities.

IMPRESSION: No acute cardiopulmonary abnormalities.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 2471, 1315, 3910

---

### study_id=360  Δtoken_F1=+0.030

**Image**: `data\openi\images\360__CXR360_IM-1776-1001.png`

**Reference**:

```
IMPRESSION: Heart size is normal. Lungs are clear. No effusion, nodules, adenopathy, or masses.
```

**RAG prediction**:

```
INDICATION: XXXX XXXX XXXX with chronic XXXX

FINDINGS: Heart size, mediastinal and aortic contours are within normal limits. Normal pulmonary vasculature. The lungs are clear. No visible pneumothorax or large pleural effusion. Elevated right hemidiaphragm. Mild degenerative changes of the spine.

IMPRESSION: 1. Elevated right hemidiaphragm. 2. No evidence of active disease.
```

**Baseline prediction**:

```
FINDINGS: No abnormality identified.
IMPRESSION: Normal chest radiograph.
```

**Top retrieved study_ids**: 3110, 1897, 1010

---

