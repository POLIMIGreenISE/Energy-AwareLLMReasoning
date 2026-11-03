# Logit-confidence probe router (step 3c)

N=1710. Probe cost: drafts 24 + logits 5.9 mWh.

Diagnostics: P(true)->no_think-correct AUC 0.828; solvability-among-hard AUC 0.461; p_need easy/hard AUC 0.995

|policy|label AUC|EAA-AUC (honest)|
|--|--|--|
|A text-only|0.900|0.9002|
|B text+drafts|0.899|0.8990|
|D text+drafts+probes|0.908|0.8996|
|E drafts+probes only|0.875|0.8889|
|R p_true rule (no training)|0.810|0.8721|
|P p_need rule (no training)|0.854|0.8737|
|oracle frontier||0.9290|

p_need rule = think iff p_need_think > tau (single forward-pass difficulty probe, no drafts, no embedding, no training; overhead probe_wh/4 ~ 1.5 mWh/q). Completes the strategy comparison: learned router vs draft-agreement rule vs self-assessment rules.
