# Generalization eval: OTB-trained routers on mohseni (N=1810)

Never 0.611 @ 701 mWh; Always 0.823 @ 4237 mWh; Oracle EAA 0.8561. Judge: reference-free grade-both (gemini-2.5-flash, certified).

**Random-router EAA 0.7164 (43% of oracle gain)** — the floor any real router must beat (Always ≫ Never here, so random routing already captures gain).

|router (trained on OTB)|label-AUC vs our oracle|EAA on new data|% oracle gain|vs random|
|--|--|--|--|--|
|LogReg+MiniLM|0.505|0.7304|49%|+6pp|
|ModernBERT|0.521|0.7301|49%|+6pp|
|p_need|0.564|0.6572|19%|-24pp|

Our energy-aware oracle vs Mohseni's own labels: **agreement 0.554** (our think 27% vs his 50%). By source:

- AIME-1983-2024       agree 0.591 (n=22)
- Big-Math-RL          agree 0.559 (n=696)
- Nectar               agree 0.583 (n=593)
- WildChat-filtered    agree 0.509 (n=499)
