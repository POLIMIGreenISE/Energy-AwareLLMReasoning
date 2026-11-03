# E2 — Does the token cost proxy mislead an energy-budgeted controller?

Measured J/token spans **42x** across batch (1→64): {1: 3.422, 2: 1.702, 4: 0.865, 8: 0.454, 16: 0.244, 32: 0.134, 64: 0.082}. The controller's *belief* about cost is the only thing that differs; the physics (measured power integrated over time) is identical.

## E2a — open-loop: the controller plans with its own cost model

|regime|budget|cost model|actual vs budget|accuracy|
|--|--|--|--|--|
|bimodal|35%|token-blind J(batch=1)|-41.6%|0.849|
|bimodal|35%|token-blind J(batch=8)|+115.8%|0.926|
|bimodal|35%|token-blind J(batch=32)|+115.8%|0.926|
|bimodal|35%|load-aware|-0.8%|0.890|
|bimodal|60%|token-blind J(batch=1)|-43.8%|0.869|
|bimodal|60%|token-blind J(batch=8)|+51.5%|0.926|
|bimodal|60%|token-blind J(batch=32)|+51.5%|0.926|
|bimodal|60%|load-aware|-1.0%|0.915|
|stable|35%|token-blind J(batch=1)|-28.8%|0.834|
|stable|35%|token-blind J(batch=8)|+5.7%|0.875|
|stable|35%|token-blind J(batch=32)|+27.6%|0.923|
|stable|35%|load-aware|+0.8%|0.865|
|stable|60%|token-blind J(batch=1)|-39.2%|0.834|
|stable|60%|token-blind J(batch=8)|-7.0%|0.884|
|stable|60%|token-blind J(batch=32)|+11.8%|0.925|
|stable|60%|load-aware|+2.6%|0.902|

## E2b — closed-loop: both meter real energy and land on budget; compare accuracy

Both controllers run on the SAME traces, so the CI is on the **per-seed paired difference** over 12 traces (an unpaired CI would drown the effect in trace-to-trace noise).

|regime|budget|router|token-blind acc|load-aware acc|paired delta (pp)|95% CI|positive traces|
|--|--|--|--|--|--|--|--|
|bimodal|20%|ModernBERT|0.8462|0.8465|+0.03|±0.15|33%|
|bimodal|20%|random|0.8456|0.8460|+0.05|±0.21|25%|
|bimodal|35%|ModernBERT|0.8615|0.8661|+0.47|±0.35|92%|
|bimodal|35%|random|0.8599|0.8610|+0.10|±0.20|42%|
|bimodal|50%|ModernBERT|0.8768|0.8799|+0.31|±0.24|67%|
|bimodal|50%|random|0.8715|0.8749|+0.33|±0.27|67%|
|bimodal|65%|ModernBERT|0.8872|0.8916|+0.44|±0.33|67%|
|bimodal|65%|random|0.8831|0.8852|+0.22|±0.24|58%|
|bimodal|80%|ModernBERT|0.9042|0.9078|+0.37|±0.35|75%|
|bimodal|80%|random|0.8978|0.9005|+0.26|±0.26|67%|
|stable|20%|ModernBERT|0.8749|0.8746|-0.03|±0.10|50%|
|stable|20%|random|0.8636|0.8652|+0.16|±0.15|67%|
|stable|35%|ModernBERT|0.8930|0.8931|+0.01|±0.11|17%|
|stable|35%|random|0.8881|0.8893|+0.12|±0.11|50%|
|stable|50%|ModernBERT|0.9104|0.9101|-0.03|±0.14|25%|
|stable|50%|random|0.9049|0.9050|+0.01|±0.20|42%|
|stable|65%|ModernBERT|0.9151|0.9150|-0.01|±0.08|25%|
|stable|65%|random|0.9143|0.9142|-0.01|±0.05|25%|
|stable|80%|ModernBERT|0.9183|0.9182|-0.01|±0.03|8%|
|stable|80%|random|0.9180|0.9175|-0.05|±0.05|8%|
