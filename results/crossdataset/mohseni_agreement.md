# Cross-dataset generalization: OTB router vs Mohseni labels (decision agreement)

OTB router (LogReg+MiniLM, trained on all 1710 OTB queries) predicting on Mohseni's 1812 held-out test prompts. His labels use a DIFFERENT definition (his models, not our energy-aware Qwen3-8B), so this is decision AGREEMENT, a directional generalization signal -- not accuracy.

|subset|n|AUC|agree@0.5|balanced acc|his think%|
|--|--|--|--|--|--|
|ALL|1812|0.504|0.510|0.510|50|
|MATH (Big-Math+AIME)|719|0.599|0.549|0.564|57|
|CHAT (Nectar+WildChat)|1093|0.495|0.484|0.499|45|
|Big-Math-RL|697|0.598|0.549|0.563|57|
|AIME-1983-2024|22|0.615|0.545|0.531|73|
|Nectar|594|0.519|0.453|0.517|42|
|WildChat-filtered|499|0.516|0.521|0.519|49|

Overall AUC 0.504; best threshold-swept agreement 0.515 @ tau=0.26.

**Reading:** AUC is the calibration-free signal (does our router RANK Mohseni's think-queries above his no_think ones?). agree@0.5 also depends on our OTB prior (88% no_think) transferring, which mismatches Mohseni's balanced 50/50 mix -- so the threshold-swept best agreement is the fairer point estimate of the transferable signal.
