# E11 x E8b: production arrival traces under a tail-latency target

E11's protocol exactly (same traces, 12 windows x 500 queries, same grids, same selection), with E8b's p95 constraint added: at each energy cap, each policy's best operating point must meet the cap **and** the p95 target. Targets are multiples of the never-reasoning p95 of that arrival process. `*` marks a 95% paired bootstrap interval that excludes zero. The unconstrained rows reproduce `REAL_TRACE.md` exactly (checked in the script).

## Never-reasoning p95 by arrival process

|arrival process|never-reasoning p95|1.5x|3x|6x|
|--|--|--|--|--|
|Poisson @0.8 (control)|14.0 s|21 s|42 s|84 s|
|Azure conv @0.8|14.7 s|22 s|44 s|88 s|
|Azure code @0.8|16.4 s|25 s|49 s|98 s|
|Azure code @0.8 (thinned)|16.1 s|24 s|48 s|96 s|

## Energy cap 25%

|arrival process|p95 target|target|static acc|static p95|static reasons|load-aware acc|load-aware p95|load-aware reasons|gap (pp), 95% CI|
|--|--|--|--|--|--|--|--|--|--|
|Poisson @0.8 (control)|none|-|0.8427|16 s|1.4%|0.8422|25 s|1.7%|-0.05 [-0.83, +0.90]|
|Poisson @0.8 (control)|1.5x|21 s|0.8427|16 s|1.4%|0.8391|21 s|1.2%|-0.37 [-0.96, +0.45]|
|Poisson @0.8 (control)|3x|42 s|0.8427|16 s|1.4%|0.8422|25 s|1.7%|-0.05 [-0.83, +0.90]|
|Poisson @0.8 (control)|6x|84 s|0.8427|16 s|1.4%|0.8422|25 s|1.7%|-0.05 [-0.83, +0.90]|
|Azure conv @0.8|none|-|0.8542|20 s|2.8%|0.8531|43 s|3.2%|-0.11 [-1.53, +1.55]|
|Azure conv @0.8|1.5x|22 s|0.8542|20 s|2.8%|0.8383|18 s|0.6%|-1.58 [-2.18, -0.86] *|
|Azure conv @0.8|3x|44 s|0.8542|20 s|2.8%|0.8531|43 s|3.2%|-0.11 [-1.53, +1.55]|
|Azure conv @0.8|6x|88 s|0.8542|20 s|2.8%|0.8531|43 s|3.2%|-0.11 [-1.53, +1.55]|
|Azure code @0.8|none|-|0.8458|19 s|1.5%|0.8562|44 s|3.7%|+1.03 [-0.13, +2.44]|
|Azure code @0.8|1.5x|25 s|0.8458|19 s|1.5%|0.8439|24 s|1.6%|-0.19 [-0.93, +0.86]|
|Azure code @0.8|3x|49 s|0.8458|19 s|1.5%|0.8562|44 s|3.7%|+1.03 [-0.13, +2.44]|
|Azure code @0.8|6x|98 s|0.8458|19 s|1.5%|0.8562|44 s|3.7%|+1.03 [-0.13, +2.44]|
|Azure code @0.8 (thinned)|none|-|0.8406|18 s|1.1%|0.8457|34 s|2.1%|+0.52 [-0.37, +1.56]|
|Azure code @0.8 (thinned)|1.5x|24 s|0.8406|18 s|1.1%|0.8424|23 s|1.5%|+0.18 [-0.52, +1.03]|
|Azure code @0.8 (thinned)|3x|48 s|0.8406|18 s|1.1%|0.8457|34 s|2.1%|+0.52 [-0.37, +1.56]|
|Azure code @0.8 (thinned)|6x|96 s|0.8406|18 s|1.1%|0.8457|34 s|2.1%|+0.52 [-0.37, +1.56]|

## Energy cap 40%

|arrival process|p95 target|target|static acc|static p95|static reasons|load-aware acc|load-aware p95|load-aware reasons|gap (pp), 95% CI|
|--|--|--|--|--|--|--|--|--|--|
|Poisson @0.8 (control)|none|-|0.8456|16 s|1.9%|0.8474|31 s|2.6%|+0.18 [-0.75, +1.22]|
|Poisson @0.8 (control)|1.5x|21 s|0.8456|16 s|1.9%|0.8391|21 s|1.2%|-0.65 [-1.27, +0.20]|
|Poisson @0.8 (control)|3x|42 s|0.8456|16 s|1.9%|0.8474|31 s|2.6%|+0.18 [-0.75, +1.22]|
|Poisson @0.8 (control)|6x|84 s|0.8456|16 s|1.9%|0.8474|31 s|2.6%|+0.18 [-0.75, +1.22]|
|Azure conv @0.8|none|-|0.8682|33 s|5.1%|0.8587|50 s|4.2%|-0.95 [-2.43, +0.69]|
|Azure conv @0.8|1.5x|22 s|0.8575|22 s|3.4%|0.8383|18 s|0.6%|-1.92 [-2.52, -1.16] *|
|Azure conv @0.8|3x|44 s|0.8682|33 s|5.1%|0.8545|43 s|3.4%|-1.37 [-2.78, +0.26]|
|Azure conv @0.8|6x|88 s|0.8682|33 s|5.1%|0.8587|50 s|4.2%|-0.95 [-2.43, +0.69]|
|Azure code @0.8|none|-|0.8519|21 s|2.4%|0.8761|86 s|7.2%|+2.42 [+1.00, +3.86] *|
|Azure code @0.8|1.5x|25 s|0.8519|21 s|2.4%|0.8439|24 s|1.6%|-0.80 [-1.58, +0.27]|
|Azure code @0.8|3x|49 s|0.8519|21 s|2.4%|0.8591|48 s|4.2%|+0.72 [-0.40, +2.03]|
|Azure code @0.8|6x|98 s|0.8519|21 s|2.4%|0.8761|86 s|7.2%|+2.42 [+1.00, +3.86] *|
|Azure code @0.8 (thinned)|none|-|0.8448|19 s|1.8%|0.8586|51 s|4.3%|+1.38 [+0.05, +2.93] *|
|Azure code @0.8 (thinned)|1.5x|24 s|0.8448|19 s|1.8%|0.8424|23 s|1.5%|-0.24 [-1.02, +0.68]|
|Azure code @0.8 (thinned)|3x|48 s|0.8448|19 s|1.8%|0.8569|48 s|4.0%|+1.21 [-0.08, +2.78]|
|Azure code @0.8 (thinned)|6x|96 s|0.8448|19 s|1.8%|0.8586|51 s|4.3%|+1.38 [+0.05, +2.93] *|

## Energy cap 55%

|arrival process|p95 target|target|static acc|static p95|static reasons|load-aware acc|load-aware p95|load-aware reasons|gap (pp), 95% CI|
|--|--|--|--|--|--|--|--|--|--|
|Poisson @0.8 (control)|none|-|0.8500|18 s|2.9%|0.8525|41 s|3.8%|+0.25 [-0.92, +1.47]|
|Poisson @0.8 (control)|1.5x|21 s|0.8500|18 s|2.9%|0.8391|21 s|1.2%|-1.09 [-1.78, -0.22] *|
|Poisson @0.8 (control)|3x|42 s|0.8500|18 s|2.9%|0.8525|41 s|3.8%|+0.25 [-0.92, +1.47]|
|Poisson @0.8 (control)|6x|84 s|0.8500|18 s|2.9%|0.8525|41 s|3.8%|+0.25 [-0.92, +1.47]|
|Azure conv @0.8|none|-|0.8824|76 s|7.2%|0.8732|78 s|6.8%|-0.92 [-2.50, +0.65]|
|Azure conv @0.8|1.5x|22 s|0.8575|22 s|3.4%|0.8383|18 s|0.6%|-1.92 [-2.52, -1.16] *|
|Azure conv @0.8|3x|44 s|0.8720|42 s|5.7%|0.8545|43 s|3.4%|-1.75 [-3.20, -0.07] *|
|Azure conv @0.8|6x|88 s|0.8824|76 s|7.2%|0.8732|78 s|6.8%|-0.92 [-2.50, +0.65]|
|Azure code @0.8|none|-|0.8677|35 s|5.0%|0.8874|113 s|9.4%|+1.97 [+0.52, +3.27] *|
|Azure code @0.8|1.5x|25 s|0.8583|24 s|3.5%|0.8439|24 s|1.6%|-1.44 [-2.24, -0.42] *|
|Azure code @0.8|3x|49 s|0.8677|35 s|5.0%|0.8591|48 s|4.2%|-0.87 [-2.01, +0.41]|
|Azure code @0.8|6x|98 s|0.8677|35 s|5.0%|0.8814|98 s|8.3%|+1.37 [-0.08, +2.73]|
|Azure code @0.8 (thinned)|none|-|0.8513|21 s|2.7%|0.8737|76 s|7.0%|+2.23 [+1.01, +3.61] *|
|Azure code @0.8 (thinned)|1.5x|24 s|0.8513|21 s|2.7%|0.8424|23 s|1.5%|-0.89 [-1.70, +0.04]|
|Azure code @0.8 (thinned)|3x|48 s|0.8513|21 s|2.7%|0.8569|48 s|4.0%|+0.56 [-0.73, +2.11]|
|Azure code @0.8 (thinned)|6x|96 s|0.8513|21 s|2.7%|0.8737|76 s|7.0%|+2.23 [+1.01, +3.61] *|

## Energy cap 70%

|arrival process|p95 target|target|static acc|static p95|static reasons|load-aware acc|load-aware p95|load-aware reasons|gap (pp), 95% CI|
|--|--|--|--|--|--|--|--|--|--|
|Poisson @0.8 (control)|none|-|0.8789|67 s|7.0%|0.8733|84 s|7.9%|-0.56 [-1.66, +0.54]|
|Poisson @0.8 (control)|1.5x|21 s|0.8541|21 s|3.5%|0.8391|21 s|1.2%|-1.50 [-2.17, -0.70] *|
|Poisson @0.8 (control)|3x|42 s|0.8693|42 s|5.6%|0.8543|42 s|4.2%|-1.51 [-2.62, -0.38] *|
|Poisson @0.8 (control)|6x|84 s|0.8789|67 s|7.0%|0.8733|84 s|7.9%|-0.56 [-1.66, +0.54]|
|Azure conv @0.8|none|-|0.9215|158 s|15.4%|0.9017|142 s|12.8%|-1.98 [-3.48, -0.88] *|
|Azure conv @0.8|1.5x|22 s|0.8575|22 s|3.4%|0.8383|18 s|0.6%|-1.92 [-2.52, -1.16] *|
|Azure conv @0.8|3x|44 s|0.8720|42 s|5.7%|0.8545|43 s|3.4%|-1.75 [-3.20, -0.07] *|
|Azure conv @0.8|6x|88 s|0.8849|86 s|7.6%|0.8782|85 s|7.7%|-0.68 [-2.25, +0.89]|
|Azure code @0.8|none|-|0.9122|158 s|12.5%|0.9115|170 s|14.1%|-0.08 [-0.73, +0.47]|
|Azure code @0.8|1.5x|25 s|0.8583|24 s|3.5%|0.8439|24 s|1.6%|-1.44 [-2.24, -0.42] *|
|Azure code @0.8|3x|49 s|0.8733|48 s|5.8%|0.8591|48 s|4.2%|-1.43 [-2.60, -0.09] *|
|Azure code @0.8|6x|98 s|0.8861|97 s|8.0%|0.8814|98 s|8.3%|-0.47 [-1.78, +0.80]|
|Azure code @0.8 (thinned)|none|-|0.8857|110 s|8.2%|0.9071|166 s|14.6%|+2.14 [+1.66, +2.66] *|
|Azure code @0.8 (thinned)|1.5x|24 s|0.8589|23 s|3.8%|0.8424|23 s|1.5%|-1.65 [-2.37, -0.82] *|
|Azure code @0.8 (thinned)|3x|48 s|0.8697|48 s|5.6%|0.8569|48 s|4.0%|-1.28 [-2.48, +0.19]|
|Azure code @0.8 (thinned)|6x|96 s|0.8823|96 s|7.6%|0.8822|96 s|8.6%|-0.01 [-1.00, +1.04]|

*Repro: `python scripts/e11_slo.py` (CPU, 286 s on this machine).*
