# JKP Run Results Analysis


## Data Sources
- Runs scanned: `/home/sh.mo/projects/justkeepprompting/outputs`
- STAR source referenced: `/home/sh.mo/projects/justkeepprompting/star_data_qa/star_clips_qa.json`


## High-Level Run Summary
- Total completed runs: **720**
- Initial accuracy: **495/720 (68.8%)**
- Final accuracy: **488/720 (67.8%)**
- Net accuracy change: **-1.0 pp**
- Improved (wrong→correct): **27 (3.8%)**
- Regressed (correct→wrong): **34 (4.7%)**
- Stable correct (correct→correct): **461 (64.0%)**
- Stable incorrect (wrong→wrong): **198 (27.5%)**

## Behavioral Dynamics (Answer Flips)
- Runs with at least one flip: **187/720 (26.0%)**
- Avg flips/run: **1.16** | Median flips/run: **0.0**
- Flip count distribution:
  - `0` flips: **533** runs
  - `1` flips: **18** runs
  - `2` flips: **45** runs
  - `3` flips: **26** runs
  - `4` flips: **34** runs
  - `5` flips: **9** runs
  - `6` flips: **11** runs
  - `7` flips: **8** runs
  - `8` flips: **4** runs
  - `9` flips: **3** runs
  - `10` flips: **29** runs

## Flip Transition Analysis
- Initial→Final correctness states:
  - `C->C`: **462 (64.2%)**
  - `C->W`: **33 (4.6%)**
  - `W->C`: **27 (3.8%)**
  - `W->W`: **198 (27.5%)**
- Runs that were ever correct at any turn: **573/720 (79.6%)**
- Runs that were ever wrong at any turn: **303/720 (42.1%)**
- Runs that became wrong after being correct (`C...W`): **152/720 (21.1%)**
- Runs that recovered to correct after being wrong (`W...C`): **143/720 (19.9%)**
- Among runs that started correct, later became wrong: **78/495 (15.8%)**
- Among runs that started wrong, recovered at least once: **78/225 (34.7%)**
- Wrong-final runs (`...->W`) decomposition:
  - Never correct at any turn (`W` only): **147/231 (63.6%)**
  - Was correct at some point then ended wrong (`...C...W`): **84/231 (36.4%)**
- Correct-final runs (`...->C`) decomposition:
  - Always correct (`C` only): **417/489 (85.3%)**
  - Recovered from wrong then ended correct (`...W...C`): **72/489 (14.7%)**
- Most common condensed correctness trajectories:
  - `C`: **417 (57.9%)**
  - `W`: **147 (20.4%)**
  - `C->W->C`: **25 (3.5%)**
  - `W->C->W`: **23 (3.2%)**
  - `W->C->W->C->W`: **17 (2.4%)**
  - `C->W`: **13 (1.8%)**
  - `C->W->C->W->C`: **12 (1.7%)**
  - `C->W->C->W`: **10 (1.4%)**
  - `W->C->W->C`: **10 (1.4%)**
  - `W->C->W->C->W->C`: **8 (1.1%)**

## Flip Transitions by Model and Prompting Type
- `Qwen/Qwen3-VL-30B-A3B-Instruct`:
  - `adversarial_negation` (**80** runs):
    - `C->C`: **54 (67.5%)**
    - `C->W`: **6 (7.5%)**
    - `W->C`: **4 (5.0%)**
    - `W->W`: **16 (20.0%)**
    - ever `C`: **75/80 (93.8%)** | ever `W`: **34/80 (42.5%)**
    - `C...W`: **29/80 (36.2%)** | `W...C`: **26/80 (32.5%)**
    - top trajectories: `C` 46 (57.5%), `W->C->W` 5 (6.2%), `C->W->C->W->C->W->C->W->C->W->C` 5 (6.2%)
  - `context_socratic` (**80** runs):
    - `C->C`: **60 (75.0%)**
    - `C->W`: **0 (0.0%)**
    - `W->C`: **0 (0.0%)**
    - `W->W`: **20 (25.0%)**
    - ever `C`: **60/80 (75.0%)** | ever `W`: **20/80 (25.0%)**
    - `C...W`: **0/80 (0.0%)** | `W...C`: **0/80 (0.0%)**
    - top trajectories: `C` 60 (75.0%), `W` 20 (25.0%)
  - `pure_socratic` (**80** runs):
    - `C->C`: **62 (77.5%)**
    - `C->W`: **0 (0.0%)**
    - `W->C`: **0 (0.0%)**
    - `W->W`: **18 (22.5%)**
    - ever `C`: **62/80 (77.5%)** | ever `W`: **18/80 (22.5%)**
    - `C...W`: **0/80 (0.0%)** | `W...C`: **0/80 (0.0%)**
    - top trajectories: `C` 62 (77.5%), `W` 18 (22.5%)
- `gemini-2.5-pro`:
  - `adversarial_negation` (**80** runs):
    - `C->C`: **48 (60.0%)**
    - `C->W`: **6 (7.5%)**
    - `W->C`: **4 (5.0%)**
    - `W->W`: **22 (27.5%)**
    - ever `C`: **72/80 (90.0%)** | ever `W`: **35/80 (43.8%)**
    - `C...W`: **27/80 (33.8%)** | `W...C`: **27/80 (33.8%)**
    - top trajectories: `C` 45 (56.2%), `W` 8 (10.0%), `W->C->W->C->W` 6 (7.5%)
  - `context_socratic` (**80** runs):
    - `C->C`: **55 (68.8%)**
    - `C->W`: **0 (0.0%)**
    - `W->C`: **0 (0.0%)**
    - `W->W`: **25 (31.2%)**
    - ever `C`: **55/80 (68.8%)** | ever `W`: **25/80 (31.2%)**
    - `C...W`: **0/80 (0.0%)** | `W...C`: **0/80 (0.0%)**
    - top trajectories: `C` 55 (68.8%), `W` 25 (31.2%)
  - `pure_socratic` (**80** runs):
    - `C->C`: **54 (67.5%)**
    - `C->W`: **1 (1.2%)**
    - `W->C`: **2 (2.5%)**
    - `W->W`: **23 (28.7%)**
    - ever `C`: **57/80 (71.2%)** | ever `W`: **27/80 (33.8%)**
    - `C...W`: **4/80 (5.0%)** | `W...C`: **3/80 (3.8%)**
    - top trajectories: `C` 53 (66.2%), `W` 23 (28.7%), `W->C->W->C` 2 (2.5%)
- `gpt-4o`:
  - `adversarial_negation` (**80** runs):
    - `C->C`: **45 (56.2%)**
    - `C->W`: **4 (5.0%)**
    - `W->C`: **4 (5.0%)**
    - `W->W`: **27 (33.8%)**
    - ever `C`: **60/80 (75.0%)** | ever `W`: **36/80 (45.0%)**
    - `C...W`: **15/80 (18.8%)** | `W...C`: **15/80 (18.8%)**
    - top trajectories: `C` 44 (55.0%), `W` 20 (25.0%), `W->C->W` 3 (3.8%)
  - `context_socratic` (**80** runs):
    - `C->C`: **45 (56.2%)**
    - `C->W`: **7 (8.8%)**
    - `W->C`: **5 (6.2%)**
    - `W->W`: **23 (28.7%)**
    - ever `C`: **64/80 (80.0%)** | ever `W`: **51/80 (63.7%)**
    - `C...W`: **33/80 (41.2%)** | `W...C`: **30/80 (37.5%)**
    - top trajectories: `C` 29 (36.2%), `W` 16 (20.0%), `C->W->C` 15 (18.8%)
  - `pure_socratic` (**80** runs):
    - `C->C`: **39 (48.8%)**
    - `C->W`: **9 (11.2%)**
    - `W->C`: **8 (10.0%)**
    - `W->W`: **24 (30.0%)**
    - ever `C`: **68/80 (85.0%)** | ever `W`: **57/80 (71.2%)**
    - `C...W`: **44/80 (55.0%)** | `W...C`: **42/80 (52.5%)**
    - top trajectories: `C` 23 (28.7%), `W` 12 (15.0%), `C->W->C` 8 (10.0%)


## Confidence Trends
- Avg initial confidence: **90.6**
- Avg final confidence: **89.2**
- Avg confidence shift (final-initial): **-1.42**

## Breakdown by Strategy
- `context_socratic`: **240** runs | initial **69.6%** | final **68.8%** | improved **5** | regressed **7** | avg flips **0.44** | avg tokens **87338** | avg time **172.8s**
- `pure_socratic`: **240** runs | initial **68.8%** | final **68.8%** | improved **10** | regressed **10** | avg flips **0.85** | avg tokens **83836** | avg time **145.1s**
- `adversarial_negation`: **240** runs | initial **67.9%** | final **65.8%** | improved **12** | regressed **17** | avg flips **2.20** | avg tokens **83754** | avg time **183.1s**

## Breakdown by Model
- `Qwen/Qwen3-VL-30B-A3B-Instruct`: **240** runs | initial **75.8%** | final **75.0%** | improved **4** | regressed **6** | avg flips **1.01** | avg tokens **50729** | avg time **142.7s**
- `gemini-2.5-pro`: **240** runs | initial **68.3%** | final **67.9%** | improved **6** | regressed **7** | avg flips **0.85** | avg tokens **166340** | avg time **213.8s**
- `gpt-4o`: **240** runs | initial **62.1%** | final **60.4%** | improved **17** | regressed **21** | avg flips **1.62** | avg tokens **37859** | avg time **144.5s**

## Breakdown by Category (Run Results)
- `Interaction`: **180** runs | initial **95.0%** | final **91.7%** | improved **2** | regressed **8** | avg flips **0.38**
- `Sequence`: **180** runs | initial **83.9%** | final **83.9%** | improved **4** | regressed **4** | avg flips **0.52**
- `Prediction`: **180** runs | initial **72.8%** | final **70.6%** | improved **5** | regressed **9** | avg flips **1.14**
- `Feasibility`: **180** runs | initial **23.3%** | final **25.0%** | improved **16** | regressed **13** | avg flips **2.62**

## Breakdown by Question Type / Template (Run Results)
- `T1`: **540** runs | initial **83.9%** | final **82.0%** | improved **11** | regressed **21** | avg flips **0.68**
- `T2`: **180** runs | initial **23.3%** | final **25.0%** | improved **16** | regressed **13** | avg flips **2.62**

## Strategy x Category Final Accuracy
- `adversarial_negation`:
  - `Feasibility`: **12/60 (20.0%)**
  - `Interaction`: **56/60 (93.3%)**
  - `Prediction`: **41/60 (68.3%)**
  - `Sequence`: **49/60 (81.7%)**
- `context_socratic`:
  - `Feasibility`: **14/60 (23.3%)**
  - `Interaction`: **55/60 (91.7%)**
  - `Prediction`: **44/60 (73.3%)**
  - `Sequence`: **52/60 (86.7%)**
- `pure_socratic`:
  - `Feasibility`: **19/60 (31.7%)**
  - `Interaction`: **54/60 (90.0%)**
  - `Prediction`: **42/60 (70.0%)**
  - `Sequence`: **50/60 (83.3%)**

## Model x Strategy Final Accuracy
- `Qwen/Qwen3-VL-30B-A3B-Instruct`:
  - `adversarial_negation`: **58/80 (72.5%)**
  - `context_socratic`: **60/80 (75.0%)**
  - `pure_socratic`: **62/80 (77.5%)**
- `gemini-2.5-pro`:
  - `adversarial_negation`: **52/80 (65.0%)**
  - `context_socratic`: **55/80 (68.8%)**
  - `pure_socratic`: **56/80 (70.0%)**
- `gpt-4o`:
  - `adversarial_negation`: **48/80 (60.0%)**
  - `context_socratic`: **50/80 (62.5%)**
  - `pure_socratic`: **47/80 (58.8%)**


## Notes
- All performance stats in this report are run-based (each JSON file is one run).
- Correctness trajectories are computed from per-turn answer letters against the run gold answer.
- `C` means correct, `W` means wrong; condensed trajectories collapse repeated same-state turns.
