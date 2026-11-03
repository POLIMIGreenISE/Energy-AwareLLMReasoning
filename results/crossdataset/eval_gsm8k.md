# Generalization eval: OTB-trained routers on gsm8k (N=1319)

Never 0.948 @ 270 mWh; Always 0.960 @ 2452 mWh; Oracle EAA 0.9746. Judge: gold.

**Random-router EAA 0.9545 (25% of oracle gain)** — the floor any real router must beat (when Always ≫ Never, random routing already captures gain).

|router (trained on OTB)|label-AUC vs our oracle|EAA on new data|% oracle gain|vs random|
|--|--|--|--|--|
|LogReg+MiniLM|0.567|0.9562|31%|+6pp|
|ModernBERT|0.623|0.9560|31%|+5pp|
|p_need|0.602|0.9515|14%|-11pp|
