# Corrected head-to-head: load-aware vs a properly swept fixed threshold

The previous grid sampled quantiles of the score VALUES and collapsed to 26 operating points; the router actually admits 684. The dense grid samples the sorted unique scores by index, reaching 174 distinct think-rates. Same traces, same queries, all 500 questions answered, accuracy read at matched energy.

```

### bimodal: never 79 Wh, always 423 Wh over 500 questions (6 identical traces per policy)
      cap |  fixed-thr (OLD grid) |  fixed-thr (DENSE grid) |          load-aware | gap vs DENSE
          |       acc         Wh |       acc  think     Wh |       acc think   Wh | 
     165W |    0.8308       81.8 |    0.8495     3%  164.3 |    0.8713   9%  157 |       +2.18pp
     217W |    0.8532      171.6 |    0.8698     6%  216.6 |    0.8852  12%  184 |       +1.53pp
     268W |    0.8823      255.2 |    0.8907     9%  268.0 |    0.8955  16%  224 |       +0.48pp
     320W |    0.9018      290.7 |    0.9138    15%  318.1 |    0.9130  21%  305 |       -0.08pp

### stable: never 29 Wh, always 56 Wh over 500 questions (6 identical traces per policy)
      cap |  fixed-thr (OLD grid) |  fixed-thr (DENSE grid) |          load-aware | gap vs DENSE
          |       acc         Wh |       acc  think     Wh |       acc think   Wh | 
      35W |    0.8308       29.4 |    0.8388     1%   35.3 |    0.8417   3%   35 |       +0.28pp
      39W |    0.8308       29.4 |    0.8393     1%   39.1 |    0.8417   3%   35 |       +0.23pp
      44W |    0.8308       29.4 |    0.8433     2%   42.5 |    0.8417   3%   35 |       -0.17pp
      48W |    0.8308       29.4 |    0.8462     3%   47.6 |    0.8705   8%   44 |       +2.43pp
```
