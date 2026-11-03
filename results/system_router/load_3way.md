# tau(load) 3-way: LogReg+MiniLM vs fine-tuned ModernBERT vs p_need

N=1710, lambda=1.0/Wh (energy-conscious). Completes results/system_router/summary.md (which only covered p_need) with the fine-tuned encoder added.

|strategy|static tau*|static acc|adaptive acc @ batch64|gain (pp)|
|--|--|--|--|--|
|LogReg+MiniLM|0.975|0.834|0.915|+8.1pp|
|Fine-tuned ModernBERT|1.000|0.834|0.917|+8.2pp|
|p_need rule|1.000|0.878|0.925|+4.7pp|
