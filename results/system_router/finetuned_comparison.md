# Fine-tuned encoder router vs existing router vs p_need

N=1710. Oracle frontier EAA: 0.9290.

|policy|label AUC|classifier/probe cost (mWh/q)|EAA-AUC|% of oracle gain|
|--|--|--|--|--|
|LogReg+MiniLM (existing)|0.899|0.0000|0.8999|69.3%|
|p_need rule|0.854|1.4640|0.8737|41.7%|
|fine-tuned mmbert_mohseni_warmstart|0.886|0.0414|0.9000|69.4%|
|fine-tuned mmbert_small|0.898|0.0414|0.9008|70.3%|
|fine-tuned modernbert|0.896|0.0934|0.9013|70.8%|

## Reading (fair EAA, corrected 2026-07-22)

Fine-tuned **ModernBERT is the best router** (0.9013), and all trained routers (~0.90) clearly
beat the training-free **p_need rule (0.8737, the worst)**. This CORRECTS an earlier bug: the
old EAA function normalized each policy over only its own energy range, silently excluding the
low-budget region where p_need has no operating points, which flattered p_need to a false 0.9103.
p_need's near-binary scores (1058 queries score exactly 0, 235 exactly 1) make it too COARSE to
serve low energy budgets -- its cheapest think-policy already spends ~1500 mWh. Trained encoders'
smooth scores can hit any budget, and win on both label AUC and EAA. A trained model beating a
zero-shot probe is the intuitive result; the earlier reversal was a metric artifact.
