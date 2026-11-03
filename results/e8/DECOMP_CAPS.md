# E8d at every cap: which ingredient earns the advantage

E8's protocol exactly; each ingredient of the rule added separately to a densely swept static threshold, at all four energy caps. `*` marks a 95% paired bootstrap interval excluding zero. The 25% and 40% rows reproduce `CONSOLIDATED.md` section 3 exactly, and the `+ live load` rows at 55% and 70% reproduce section 1 exactly (both checked in the script).

## near-idle

|cap|policy|accuracy|reasons|p95|gap vs static (pp), 95% CI|
|--|--|--|--|--|--|
|25%|static threshold (baseline)|0.8430|1.7%|16 s|baseline|
|25%|+ per-query tokens|0.8489|5.1%|23 s|+0.60 [+0.40, +0.78] *|
|25%|+ live load (deployable)|0.8459|2.8%|24 s|+0.30 [-0.17, +0.85]|
|25%|+ both|0.8487|4.8%|23 s|+0.58 [+0.27, +0.92] *|
|40%|static threshold (baseline)|0.8483|2.8%|18 s|baseline|
|40%|+ per-query tokens|0.8591|7.3%|39 s|+1.08 [+0.81, +1.34] *|
|40%|+ live load (deployable)|0.8599|5.3%|49 s|+1.16 [+0.46, +1.88] *|
|40%|+ both|0.8608|7.1%|38 s|+1.25 [+0.79, +1.75] *|
|55%|static threshold (baseline)|0.8582|4.3%|26 s|baseline|
|55%|+ per-query tokens|0.8713|9.4%|53 s|+1.32 [+1.03, +1.61] *|
|55%|+ live load (deployable)|0.8726|7.6%|70 s|+1.44 [+0.84, +2.07] *|
|55%|+ both|0.8759|10.1%|62 s|+1.78 [+1.09, +2.48] *|
|70%|static threshold (baseline)|0.8741|6.7%|62 s|baseline|
|70%|+ per-query tokens|0.8884|12.1%|70 s|+1.43 [+1.07, +1.82] *|
|70%|+ live load (deployable)|0.8918|11.5%|116 s|+1.77 [+1.18, +2.34] *|
|70%|+ both|0.8953|13.8%|94 s|+2.11 [+1.50, +2.73] *|

## moderate

|cap|policy|accuracy|reasons|p95|gap vs static (pp), 95% CI|
|--|--|--|--|--|--|
|25%|static threshold (baseline)|0.8409|1.4%|16 s|baseline|
|25%|+ per-query tokens|0.8658|8.6%|50 s|+2.49 [+2.15, +2.83] *|
|25%|+ live load (deployable)|0.8420|1.9%|31 s|+0.11 [-0.73, +1.16]|
|25%|+ both|0.8566|6.3%|33 s|+1.58 [+1.16, +2.08] *|
|40%|static threshold (baseline)|0.8464|2.4%|17 s|baseline|
|40%|+ per-query tokens|0.8829|11.3%|67 s|+3.65 [+3.23, +4.08] *|
|40%|+ live load (deployable)|0.8471|2.9%|37 s|+0.07 [-0.84, +1.14]|
|40%|+ both|0.8682|8.6%|50 s|+2.18 [+1.64, +2.76] *|
|55%|static threshold (baseline)|0.8515|3.2%|20 s|baseline|
|55%|+ per-query tokens|0.9003|13.8%|95 s|+4.89 [+4.41, +5.36] *|
|55%|+ live load (deployable)|0.8555|4.6%|50 s|+0.41 [-0.55, +1.43]|
|55%|+ both|0.8820|10.9%|72 s|+3.05 [+2.22, +3.89] *|
|70%|static threshold (baseline)|0.8788|7.6%|83 s|baseline|
|70%|+ per-query tokens|0.9112|16.1%|130 s|+3.24 [+2.90, +3.58] *|
|70%|+ live load (deployable)|0.8725|7.9%|86 s|-0.64 [-1.50, +0.21]|
|70%|+ both|0.8939|13.1%|94 s|+1.51 [+0.89, +2.08] *|

## saturated

|cap|policy|accuracy|reasons|p95|gap vs static (pp), 95% CI|
|--|--|--|--|--|--|
|25%|static threshold (baseline)|0.8420|1.6%|17 s|baseline|
|25%|+ per-query tokens|0.8914|12.4%|88 s|+4.94 [+4.40, +5.50] *|
|25%|+ live load (deployable)|0.8409|1.8%|26 s|-0.11 [-0.69, +0.57]|
|25%|+ both|0.8748|9.7%|64 s|+3.28 [+2.75, +3.83] *|
|40%|static threshold (baseline)|0.8862|8.5%|119 s|baseline|
|40%|+ per-query tokens|0.9102|15.9%|149 s|+2.40 [+2.07, +2.75] *|
|40%|+ live load (deployable)|0.8701|7.1%|82 s|-1.61 [-2.54, -0.71] *|
|40%|+ both|0.8953|13.5%|110 s|+0.91 [+0.57, +1.25] *|
|55%|static threshold (baseline)|0.9164|16.2%|199 s|baseline|
|55%|+ per-query tokens|0.9202|18.6%|189 s|+0.37 [+0.22, +0.53] *|
|55%|+ live load (deployable)|0.9126|16.0%|204 s|-0.38 [-0.55, -0.22] *|
|55%|+ both|0.9167|19.5%|198 s|+0.03 [-0.13, +0.17]|
|70%|static threshold (baseline)|0.9237|25.3%|264 s|baseline|
|70%|+ per-query tokens|0.9235|34.8%|261 s|-0.03 [-0.10, +0.06]|
|70%|+ live load (deployable)|0.9237|25.6%|263 s|+0.00 [-0.04, +0.04]|
|70%|+ both|0.9238|33.9%|261 s|+0.01 [-0.07, +0.08]|

## bursty

|cap|policy|accuracy|reasons|p95|gap vs static (pp), 95% CI|
|--|--|--|--|--|--|
|25%|static threshold (baseline)|0.8528|3.4%|20 s|baseline|
|25%|+ per-query tokens|0.8708|9.3%|52 s|+1.81 [+1.47, +2.14] *|
|25%|+ live load (deployable)|0.8727|7.9%|75 s|+1.99 [+1.54, +2.44] *|
|25%|+ both|0.8774|10.4%|65 s|+2.46 [+2.02, +2.89] *|
|40%|static threshold (baseline)|0.8738|6.6%|61 s|baseline|
|40%|+ per-query tokens|0.8928|12.5%|75 s|+1.91 [+1.53, +2.31] *|
|40%|+ live load (deployable)|0.8905|11.8%|123 s|+1.68 [+1.27, +2.11] *|
|40%|+ both|0.8974|15.9%|100 s|+2.37 [+1.91, +2.85] *|
|55%|static threshold (baseline)|0.8964|10.0%|122 s|baseline|
|55%|+ per-query tokens|0.9081|15.6%|114 s|+1.17 [+0.93, +1.44] *|
|55%|+ live load (deployable)|0.9047|14.2%|145 s|+0.83 [+0.64, +1.02] *|
|55%|+ both|0.9115|19.6%|131 s|+1.52 [+1.22, +1.82] *|
|70%|static threshold (baseline)|0.9159|15.7%|143 s|baseline|
|70%|+ per-query tokens|0.9202|18.6%|141 s|+0.43 [+0.30, +0.56] *|
|70%|+ live load (deployable)|0.9177|18.2%|158 s|+0.18 [+0.03, +0.32] *|
|70%|+ both|0.9197|23.4%|156 s|+0.38 [+0.24, +0.52] *|

*Repro: `python scripts/e8_decomp_caps.py` (CPU, 315 s on this machine).*
