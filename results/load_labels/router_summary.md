# Step 2 — load-adaptive threshold router

Same trained classifier P(think); only the threshold adapts: tau*(L) = argmax acc - lambda*E(L).

## lambda=0.04/Wh (accuracy-first) (static tau*=0.78)

|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|
|--|--|--|--|--|--|--|--|
|1|0.78|13%|0.909|0.909|+0.0pp|1257.8|1257.8|
|2|0.66|17%|0.920|0.909|+1.1pp|789.7|625.7|
|4|0.14|51%|0.927|0.909|+1.9pp|551.5|317.9|
|8|0.14|51%|0.927|0.909|+1.9pp|289.3|166.7|
|16|0.14|51%|0.927|0.909|+1.9pp|155.4|89.6|
|32|0.14|51%|0.927|0.909|+1.9pp|85.4|49.2|
|64|0.02|99%|0.928|0.909|+1.9pp|61.6|30.2|

## lambda=1.0/Wh (energy-conscious) (static tau*=0.97)

|load|tau(L)|think% adaptive|acc adaptive|acc static|gap|mWh adaptive|mWh static|
|--|--|--|--|--|--|--|--|
|1|0.97|0%|0.834|0.834|+0.0pp|315.0|315.0|
|2|0.97|0%|0.834|0.834|+0.0pp|156.7|156.7|
|4|0.97|0%|0.834|0.834|+0.0pp|79.6|79.6|
|8|0.96|0%|0.836|0.834|+0.2pp|43.4|41.8|
|16|0.85|9%|0.892|0.834|+5.8pp|72.2|22.4|
|32|0.72|15%|0.916|0.834|+8.2pp|56.7|12.3|
|64|0.70|16%|0.918|0.834|+8.4pp|35.9|7.6|

## Budget framing (energy-conscious static tau)

|budget mWh/q|batch 1 adaptive/static|batch 8 adaptive/static|batch 64 adaptive/static|
|--|--|--|--|
|50|-- / --|0.838 / 0.834|0.925 / 0.834|
|100|-- / --|0.870 / 0.834|0.928 / 0.834|
|200|-- / --|0.918 / 0.834|0.928 / 0.834|
|500|0.850 / 0.834|0.928 / 0.834|0.928 / 0.834|
|1000|0.888 / 0.834|0.928 / 0.834|0.928 / 0.834|
|2000|0.924 / 0.834|0.928 / 0.834|0.928 / 0.834|
