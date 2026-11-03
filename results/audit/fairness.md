# Fairness audit of the load-aware vs fixed-threshold comparison

```
## A/B. Same traffic and same questions?
  identical arrival times & queries passed to both policies: True (same object)
  completed requests: static 500/500, load-aware 500/500  -> PASS
  same query set answered: PASS by construction (the trace fixes qid order)

## C. Is the fixed threshold given its best possible setting?
  distinct router scores: 683 over 1710 queries
  distinct think-rates a fixed threshold can produce: 684
    reachable rates (first 12): [0.0, 0.001, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.005, 0.006, 0.006, 0.007]
  think-rates reachable by the SWEPT grid: 26
  -> grid covers a SUBSET of the achievable operating points
  NOTE: the router's scores are saturated (81.5% <0.1, 10.6% >0.9), so the number of achievable operating points is a
        PROPERTY OF THE ROUTER, not of our grid. A finer grid cannot create new ones.

## E. Does the traffic exercise the load range the claim depends on?
  regime                        mean L   p50    p95   max  % time L>=16
  bimodal (used in E1/E3)          6.2     3     19    27         12.1%
  stable (used in E1/E3)          20.3    20     32    43         71.8%
  high-rate (E4 sweep tail)       43.6    47     47    47         96.5%
  (L observed at admission; the J/token curve spans batch 1-64, so a regime that never
   reaches high L exercises only part of the 42x range -- the advantage is then UNDER-stated.)

## D/F. Head-to-head: same trace, same questions, matched energy, final numbers
  never-think 79 Wh, always-think 423 Wh over 500 questions

   energy cap |               BEST fixed threshold |                     load-aware
              |      acc   think     tau       Wh |      acc   think       Wh
         165W |   0.8308      0%   0.998     81.8 |   0.8665      8%    146.0   -> +3.57pp
         217W |   0.8532      4%   0.975    171.6 |   0.8867     13%    189.1   -> +3.35pp
         268W |   0.8823      8%   0.946    255.2 |   0.8867     13%    189.1   -> +0.43pp
         320W |   0.9018     11%   0.846    290.7 |   0.9092     19%    273.3   -> +0.73pp

  Both columns are the BEST feasible operating point of that policy family at that cap,
  averaged over the SAME 6 arrival traces, with all 500 questions answered in every run.
```
