# Decomposing OTB's routability: real signal or topic routing?

Joined 1710 OTB items with their own `domain`/`answer_type` metadata (HF parquet).

## Composition
```
overthinkingbench      n=1327  think  2.0%  types {np.str_('mcq'): 298, np.str_('open-ended'): 355, np.str_('numeric'): 312, np.str_('open-ended-long'): 362}
underthinkingbench     n= 383  think 45.7%  types {np.str_('unknown'): 383}
```

## How much of the AUC is topic?
```
'is it the under-half?' 1-feature router : AUC 0.864
'which domain?' lookup table            : AUC 0.857
'which answer-type?' lookup table       : AUC 0.858
FULL text router (MiniLM+LogReg)        : AUC 0.895
```

## Signal beyond topic (within-half text routing)
```
within overthinkingbench      n=1327 think  2.0% -> AUC 0.512
within underthinkingbench     n= 383 think 45.7% -> AUC 0.759
```

## Easy math in OTB
```
over NUMERIC (easy math)   n= 312 oracle think   3.2%  router mean p 0.061
under (reasoning puzzles)  n= 383 oracle think  45.7%  router mean p 0.392
```