# E2 rebuilt: holding a budget without knowing each query's token cost

E2's open-loop plan-then-deploy procedure in the shared simulator on the four E8 regimes. Budget = never + f x (always - never). lambda is bisected on 5 historical traces and frozen; the table gives energy actually spent on the 20 evaluation traces against the budget (mean, %). Token models: **true** per-query counts (as in the original E2), **constant** E[dtok] = 2370 (the shipped rule), **estimated** out-of-fold text prediction (results/e10). Anchors reproduce CONSOLIDATED.md section 0.

## Budget 35% of the span, true token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|125.9|-20|-14|+4|+17|+26|+32|+34|+37|+37|1.7 (+3.5)|**-0.0%**|40%|
|moderate|45.8|-13|-13|-13|-13|-11|-5|+1|+17|+23|0.454 (+0.8)|**-1.2%**|50%|
|saturated|24.4|-51|-51|-51|-51|-51|-51|-46|-23|+21|0.13 (+20.8)|**+3.4%**|60%|
|bursty|199.0|-45|-19|+5|+37|+70|+99|+109|+109|+109|1.7 (+5.0)|**-4.7%**|20%|

## Budget 35% of the span, constant token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|125.9|-20|-14|+7|+18|+27|+32|+35|+37|+37|1.7 (+7.4)|**+0.3%**|55%|
|moderate|45.8|-13|-13|-13|-13|-10|+8|+11|+17|+23|0.6 (+8.2)|**-8.1%**|20%|
|saturated|24.4|-51|-51|-51|-51|-51|-51|-32|+1|+21|0.244 (+1.2)|**+1.7%**|85%|
|bursty|199.0|-44|-11|+13|+41|+71|+99|+109|+109|+109|2.4 (-11.4)|**-5.4%**|25%|

## Budget 35% of the span, estimated token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|125.9|-20|-13|+3|+17|+26|+32|+35|+37|+37|1.7 (+3.5)|**-0.2%**|50%|
|moderate|45.8|-13|-13|-13|-13|-10|-3|+4|+17|+23|0.6 (-2.6)|**-1.6%**|40%|
|saturated|24.4|-51|-51|-51|-51|-51|-51|-41|-8|+21|0.244 (-8.0)|**+1.7%**|75%|
|bursty|199.0|-46|-18|+7|+36|+70|+99|+109|+109|+109|1.7 (+6.9)|**-3.8%**|20%|

## Budget 60% of the span, true token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|143.9|-30|-19|-3|+5|+13|+16|+19|+20|+20|1.7 (-3.4)|**-1.0%**|35%|
|moderate|50.0|-20|-20|-20|-20|-17|-11|-6|+7|+13|0.454 (-5.9)|**-1.0%**|35%|
|saturated|33.2|-64|-64|-64|-64|-64|-59|-53|-33|+5|0.13 (+5.2)|**+0.3%**|45%|
|bursty|282.9|-43|-27|-4|+21|+39|+47|+47|+47|+47|1.7 (-3.9)|**-3.0%**|30%|

## Budget 60% of the span, constant token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|143.9|-30|-17|-2|+7|+13|+17|+19|+20|+20|1.7 (-1.7)|**+2.3%**|65%|
|moderate|50.0|-20|-20|-20|-20|-11|-0|+2|+8|+13|0.6 (-0.2)|**+4.5%**|85%|
|saturated|33.2|-64|-64|-64|-64|-64|-41|-29|-21|+6|0.13 (+5.8)|**-1.0%**|45%|
|bursty|282.9|-38|-20|-1|+21|+39|+47|+47|+47|+47|1.7 (-0.6)|**+1.0%**|50%|

## Budget 60% of the span, estimated token counts

|regime|budget (Wh)|C=3.42|C=2.4|C=1.7|C=1.2|C=0.865|C=0.6|C=0.454|C=0.244|C=0.13|C closest|load-aware|breach|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|near-idle|143.9|-30|-19|-3|+6|+12|+17|+19|+20|+20|1.7 (-2.8)|**+1.3%**|40%|
|moderate|50.0|-20|-20|-20|-20|-17|-7|-3|+8|+13|0.454 (-3.3)|**+1.6%**|60%|
|saturated|33.2|-64|-64|-64|-64|-64|-56|-52|-27|+6|0.13 (+5.5)|**-0.3%**|50%|
|bursty|282.9|-42|-24|-3|+21|+39|+47|+47|+47|+47|1.7 (-3.2)|**-0.5%**|30%|

## Load-aware controller only: effect of the token model

|budget|regime|true tokens|constant|estimated|
|--|--|--|--|--|
|35%|near-idle|-0.0% (acc 0.8555, reasons 6.2%)|+0.3% (acc 0.8576, reasons 4.9%)|-0.2% (acc 0.8522, reasons 5.4%)|
|35%|moderate|-1.2% (acc 0.8645, reasons 7.8%)|-8.1% (acc 0.8341, reasons 0.6%)|-1.6% (acc 0.8535, reasons 5.6%)|
|35%|saturated|+3.4% (acc 0.8902, reasons 12.6%)|+1.7% (acc 0.8627, reasons 5.6%)|+1.7% (acc 0.8764, reasons 10.2%)|
|35%|bursty|-4.7% (acc 0.8883, reasons 13.3%)|-5.4% (acc 0.8834, reasons 10.5%)|-3.8% (acc 0.8858, reasons 12.5%)|
|60%|near-idle|-1.0% (acc 0.8795, reasons 10.7%)|+2.3% (acc 0.8851, reasons 9.9%)|+1.3% (acc 0.8835, reasons 11.5%)|
|60%|moderate|-1.0% (acc 0.8833, reasons 11.2%)|+4.5% (acc 0.8741, reasons 8.2%)|+1.6% (acc 0.8775, reasons 9.8%)|
|60%|saturated|+0.3% (acc 0.9201, reasons 22.7%)|-1.0% (acc 0.9174, reasons 17.9%)|-0.3% (acc 0.9186, reasons 21.3%)|
|60%|bursty|-3.0% (acc 0.9135, reasons 20.1%)|+1.0% (acc 0.9093, reasons 15.1%)|-0.5% (acc 0.9133, reasons 19.8%)|

*Repro: `python scripts/e2_transfer_tokens.py` (CPU, 164 s on this machine).*
