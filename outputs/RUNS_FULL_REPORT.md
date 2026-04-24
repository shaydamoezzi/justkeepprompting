# JKP Run Inventory and Results Analysis

_Generated on 2026-04-24 13:25:56 UTC_

## Data Sources
- Runs scanned: `/home/sh.mo/projects/justkeepprompting/outputs`
- STAR source referenced: `/home/sh.mo/projects/justkeepprompting/star_data_qa/star_clips_qa.json`
- JSON files discovered under `outputs`: **720**
- Valid run JSON files parsed: **720**
- Non-run/ignored JSON files: **0**

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

## Flip Transition Deep-Dive
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

## Confidence Score Trends (Overall)
- Avg initial confidence: **90.6** | Avg final confidence: **89.2**
- Avg confidence delta (final-initial): **-1.42**
- Avg confidence on correct turns: **91.71**
- Avg confidence on wrong turns: **85.72**
- Confidence gap (correct - wrong): **+5.99**
- Median confidence delta: **+0.00**
- Avg confidence slope per turn: **-0.121**
- Per-turn mean confidence (T0-T9): `T0:90.6, T1:89.6, T2:89.8, T3:89.7, T4:90.2, T5:90.2, T6:89.8, T7:89.3, T8:89.5, T9:89.3`

## Confidence by Correctness Pattern (Overall)
- `always_correct`: **417** runs | avg init **92.64** | avg final **93.15** | avg delta **+0.52** | median delta **+0.00** | avg slope/turn **+0.052**
  - avg conf on correct turns: **93.44**
  - avg conf on wrong turns: **NA**
  - confidence gap C-W: **NA**
  - per-turn mean confidence (T0-T9): `T0:92.6, T1:92.6, T2:93.5, T3:93.8, T4:93.9, T5:93.9, T6:93.8, T7:93.7, T8:93.5, T9:93.4`
- `always_wrong`: **147** runs | avg init **88.64** | avg final **87.91** | avg delta **-0.44** | median delta **+0.00** | avg slope/turn **-0.044**
  - avg conf on correct turns: **NA**
  - avg conf on wrong turns: **88.38**
  - confidence gap C-W: **NA**
  - per-turn mean confidence (T0-T9): `T0:88.6, T1:88.4, T2:88.9, T3:88.7, T4:88.6, T5:88.7, T6:88.4, T7:88.1, T8:88.0, T9:87.8`
- `wrong_to_correct`: **4** runs | avg init **90.00** | avg final **80.00** | avg delta **-10.00** | median delta **-10.00** | avg slope/turn **-1.000**
  - avg conf on correct turns: **82.31**
  - avg conf on wrong turns: **90.00**
  - confidence gap C-W: **-7.69**
  - per-turn mean confidence (T0-T9): `T0:90.0, T1:87.5, T2:87.5, T3:85.0, T4:85.0, T5:85.0, T6:85.0, T7:85.0, T8:85.0, T9:85.0`
- `correct_to_wrong`: **13** runs | avg init **91.15** | avg final **84.92** | avg delta **-6.23** | median delta **-5.00** | avg slope/turn **-0.623**
  - avg conf on correct turns: **92.80**
  - avg conf on wrong turns: **86.00**
  - confidence gap C-W: **+6.80**
  - per-turn mean confidence (T0-T9): `T0:91.2, T1:86.9, T2:89.8, T3:88.8, T4:90.6, T5:90.0, T6:89.6, T7:90.7, T8:89.6, T9:88.3`
- `oscillating_multi_flip`: **139** runs | avg init **86.48** | avg final **80.86** | avg delta **-5.62** | median delta **-5.00** | avg slope/turn **-0.562**
  - avg conf on correct turns: **82.38**
  - avg conf on wrong turns: **80.92**
  - confidence gap C-W: **+1.46**
  - per-turn mean confidence (T0-T9): `T0:86.5, T1:83.3, T2:81.3, T3:80.2, T4:82.2, T5:82.1, T6:81.2, T7:79.1, T8:80.6, T9:80.3`

## Turn of First Flip Analysis (Overall)
- Runs with at least one answer flip: **187/720 (26.0%)**
- Runs with no answer flip: **533/720 (74.0%)**
- Avg turn of first flip: **2.48** | Median: **1.00**
- First flip turn distribution:
  - `turn 1`: **105 (56.1% of flipped runs)**
  - `turn 2`: **23 (12.3% of flipped runs)**
  - `turn 3`: **16 (8.6% of flipped runs)**
  - `turn 4`: **15 (8.0% of flipped runs)**
  - `turn 5`: **6 (3.2% of flipped runs)**
  - `turn 6`: **6 (3.2% of flipped runs)**
  - `turn 7`: **3 (1.6% of flipped runs)**
  - `turn 8`: **4 (2.1% of flipped runs)**
  - `turn 9`: **4 (2.1% of flipped runs)**
  - `turn 10`: **5 (2.7% of flipped runs)**
- Flipped runs ending correct: avg first flip turn **2.57** (n=72)
- Flipped runs ending wrong: avg first flip turn **2.43** (n=115)

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

## Confidence and First-Flip Trends by Model and Prompting Type
- `Qwen/Qwen3-VL-30B-A3B-Instruct`:
  - `adversarial_negation` (**80** runs):
    - confidence: init **90.38**, final **94.09**, avg delta **+3.71**, median delta **+4.00**, slope/turn **+0.371**
    - confidence on states: correct **93.08**, wrong **94.68**, gap C-W **-1.60**
    - per-turn confidence (T0-T9): `T0:90.4, T1:89.7, T2:93.7, T3:93.4, T4:95.4, T5:95.1, T6:95.0, T7:93.6, T8:94.5, T9:94.2`
    - first-flip: flipped **31/80 (38.8%)**, never-flip **49/80 (61.3%)**
    - first-flip timing: avg turn **1.00**, median turn **1.00**
    - most common first-flip turns: `turn 1` 31 (100.0%)
    - pattern `always_correct`: n=46, init **91.20**, final **91.50**, delta **+0.30**, slope/turn **+0.030**
    - pattern `always_wrong`: n=5, init **85.00**, final **99.80**, delta **+14.80**, slope/turn **+1.480**
    - pattern `correct_to_wrong`: n=3, init **88.33**, final **98.00**, delta **+9.67**, slope/turn **+0.967**
    - pattern `oscillating_multi_flip`: n=26, init **90.19**, final **97.12**, delta **+6.92**, slope/turn **+0.692**
  - `context_socratic` (**80** runs):
    - confidence: init **90.88**, final **94.60**, avg delta **+3.73**, median delta **+0.00**, slope/turn **+0.372**
    - confidence on states: correct **93.28**, wrong **94.47**, gap C-W **-1.19**
    - per-turn confidence (T0-T9): `T0:90.9, T1:91.2, T2:92.7, T3:93.5, T4:94.0, T5:94.3, T6:94.4, T7:94.5, T8:94.6, T9:94.6`
    - first-flip: flipped **0/80 (0.0%)**, never-flip **80/80 (100.0%)**
    - first-flip timing: avg turn **NA**, median turn **NA**
    - pattern `always_correct`: n=60, init **91.17**, final **94.23**, delta **+3.07**, slope/turn **+0.307**
    - pattern `always_wrong`: n=20, init **90.00**, final **95.70**, delta **+5.70**, slope/turn **+0.570**
  - `pure_socratic` (**80** runs):
    - confidence: init **92.12**, final **NA**, avg delta **NA**, median delta **NA**, slope/turn **NA**
    - confidence on states: correct **92.58**, wrong **90.56**, gap C-W **+2.03**
    - per-turn confidence (T0-T9): `T0:92.1, T1:NA, T2:NA, T3:NA, T4:NA, T5:NA, T6:NA, T7:NA, T8:NA, T9:NA`
    - first-flip: flipped **1/80 (1.2%)**, never-flip **79/80 (98.8%)**
    - first-flip timing: avg turn **1.00**, median turn **1.00**
    - most common first-flip turns: `turn 1` 1 (100.0%)
    - pattern `always_correct`: n=62, init **92.58**, final **NA**, delta **NA**, slope/turn **NA**
    - pattern `always_wrong`: n=18, init **90.56**, final **NA**, delta **NA**, slope/turn **NA**
- `gemini-2.5-pro`:
  - `adversarial_negation` (**80** runs):
    - confidence: init **90.14**, final **92.56**, avg delta **+2.42**, median delta **+0.00**, slope/turn **+0.242**
    - confidence on states: correct **93.24**, wrong **83.43**, gap C-W **+9.80**
    - per-turn confidence (T0-T9): `T0:90.1, T1:89.4, T2:86.8, T3:86.4, T4:89.7, T5:91.1, T6:90.2, T7:89.8, T8:91.9, T9:91.2`
    - first-flip: flipped **28/80 (35.0%)**, never-flip **52/80 (65.0%)**
    - first-flip timing: avg turn **1.36**, median turn **1.00**
    - most common first-flip turns: `turn 1` 23 (82.1%), `turn 2` 3 (10.7%), `turn 3` 1 (3.6%)
    - pattern `always_correct`: n=45, init **94.89**, final **95.78**, delta **+0.89**, slope/turn **+0.089**
    - pattern `always_wrong`: n=8, init **90.00**, final **96.88**, delta **+6.88**, slope/turn **+0.688**
    - pattern `oscillating_multi_flip`: n=27, init **82.26**, final **85.93**, delta **+3.67**, slope/turn **+0.367**
  - `context_socratic` (**80** runs):
    - confidence: init **90.94**, final **92.10**, avg delta **+1.16**, median delta **+0.00**, slope/turn **+0.116**
    - confidence on states: correct **94.66**, wrong **85.71**, gap C-W **+8.96**
    - per-turn confidence (T0-T9): `T0:90.9, T1:91.2, T2:91.7, T3:92.0, T4:92.0, T5:92.1, T6:92.2, T7:92.2, T8:92.2, T9:92.0`
    - first-flip: flipped **1/80 (1.2%)**, never-flip **79/80 (98.8%)**
    - first-flip timing: avg turn **9.00**, median turn **9.00**
    - most common first-flip turns: `turn 9` 1 (100.0%)
    - pattern `always_correct`: n=55, init **93.09**, final **95.05**, delta **+1.96**, slope/turn **+0.196**
    - pattern `always_wrong`: n=25, init **86.20**, final **85.60**, delta **-0.60**, slope/turn **-0.060**
  - `pure_socratic` (**80** runs):
    - confidence: init **90.31**, final **92.62**, avg delta **+2.31**, median delta **+0.00**, slope/turn **+0.231**
    - confidence on states: correct **93.35**, wrong **88.05**, gap C-W **+5.30**
    - per-turn confidence (T0-T9): `T0:90.3, T1:90.2, T2:91.0, T3:91.3, T4:91.9, T5:92.2, T6:92.1, T7:92.2, T8:92.2, T9:92.5`
    - first-flip: flipped **5/80 (6.2%)**, never-flip **75/80 (93.8%)**
    - first-flip timing: avg turn **3.80**, median turn **3.00**
    - most common first-flip turns: `turn 3` 1 (20.0%), `turn 2` 1 (20.0%), `turn 1` 1 (20.0%)
    - pattern `always_correct`: n=53, init **91.98**, final **94.81**, delta **+2.83**, slope/turn **+0.283**
    - pattern `always_wrong`: n=23, init **90.00**, final **91.52**, delta **+1.52**, slope/turn **+0.152**
    - pattern `correct_to_wrong`: n=1, init **85.00**, final **40.00**, delta **-45.00**, slope/turn **-4.500**
    - pattern `oscillating_multi_flip`: n=3, init **65.00**, final **80.00**, delta **+15.00**, slope/turn **+1.500**
- `gpt-4o`:
  - `adversarial_negation` (**80** runs):
    - confidence: init **90.31**, final **89.04**, avg delta **-1.12**, median delta **+0.00**, slope/turn **-0.113**
    - confidence on states: correct **91.31**, wrong **87.01**, gap C-W **+4.31**
    - per-turn confidence (T0-T9): `T0:90.3, T1:90.5, T2:90.4, T3:90.3, T4:89.6, T5:89.4, T6:89.5, T7:89.1, T8:89.6, T9:89.6`
    - first-flip: flipped **17/80 (21.2%)**, never-flip **63/80 (78.8%)**
    - first-flip timing: avg turn **3.94**, median turn **4.00**
    - most common first-flip turns: `turn 4` 7 (41.2%), `turn 1` 3 (17.6%), `turn 3` 3 (17.6%)
    - pattern `always_correct`: n=44, init **92.27**, final **92.09**, delta **+0.00**, slope/turn **+0.000**
    - pattern `always_wrong`: n=20, init **87.50**, final **87.89**, delta **+0.50**, slope/turn **+0.050**
    - pattern `wrong_to_correct`: n=1, init **90.00**, final **90.00**, delta **+0.00**, slope/turn **+0.000**
    - pattern `correct_to_wrong`: n=1, init **95.00**, final **70.00**, delta **-25.00**, slope/turn **-2.500**
    - pattern `oscillating_multi_flip`: n=14, init **87.86**, final **82.50**, delta **-5.36**, slope/turn **-0.536**
  - `context_socratic` (**80** runs):
    - confidence: init **90.50**, final **82.00**, avg delta **-8.50**, median delta **-10.00**, slope/turn **-0.850**
    - confidence on states: correct **88.25**, wrong **80.22**, gap C-W **+8.04**
    - per-turn confidence (T0-T9): `T0:90.5, T1:88.2, T2:87.6, T3:86.2, T4:85.1, T5:85.1, T6:84.3, T7:83.5, T8:83.3, T9:82.5`
    - first-flip: flipped **47/80 (58.8%)**, never-flip **33/80 (41.2%)**
    - first-flip timing: avg turn **3.30**, median turn **2.00**
    - most common first-flip turns: `turn 1` 18 (38.3%), `turn 2` 7 (14.9%), `turn 3` 6 (12.8%)
    - pattern `always_correct`: n=29, init **94.66**, final **92.59**, delta **-2.07**, slope/turn **-0.207**
    - pattern `always_wrong`: n=16, init **88.75**, final **76.88**, delta **-11.88**, slope/turn **-1.188**
    - pattern `wrong_to_correct`: n=2, init **90.00**, final **80.00**, delta **-10.00**, slope/turn **-1.000**
    - pattern `correct_to_wrong`: n=5, init **91.00**, final **84.00**, delta **-7.00**, slope/turn **-0.700**
    - pattern `oscillating_multi_flip`: n=28, init **87.14**, final **73.75**, delta **-13.39**, slope/turn **-1.339**
  - `pure_socratic` (**80** runs):
    - confidence: init **89.75**, final **76.38**, avg delta **-13.38**, median delta **-10.00**, slope/turn **-1.337**
    - confidence on states: correct **84.97**, wrong **77.80**, gap C-W **+7.17**
    - per-turn confidence (T0-T9): `T0:89.8, T1:85.9, T2:84.7, T3:84.2, T4:83.8, T5:82.1, T6:80.9, T7:79.2, T8:77.4, T9:77.6`
    - first-flip: flipped **57/80 (71.2%)**, never-flip **23/80 (28.7%)**
    - first-flip timing: avg turn **2.53**, median turn **2.00**
    - most common first-flip turns: `turn 1` 28 (49.1%), `turn 2` 12 (21.1%), `turn 3` 5 (8.8%)
    - pattern `always_correct`: n=23, init **93.70**, final **82.83**, delta **-10.87**, slope/turn **-1.087**
    - pattern `always_wrong`: n=12, init **88.33**, final **76.67**, delta **-11.67**, slope/turn **-1.167**
    - pattern `wrong_to_correct`: n=1, init **90.00**, final **70.00**, delta **-20.00**, slope/turn **-2.000**
    - pattern `correct_to_wrong`: n=3, init **95.00**, final **93.33**, delta **-1.67**, slope/turn **-0.167**
    - pattern `oscillating_multi_flip`: n=41, init **87.56**, final **71.59**, delta **-15.98**, slope/turn **-1.598**

## Efficiency Metrics
- Avg total tokens/run: **84976.1** | Median: **46325.5**
- Avg prompt tokens/run: **80809.1**
- Avg completion tokens/run: **861.7**
- Avg wall time/run: **167.0s** | Median: **147.2s**
- Avg turns/run: **11.00** | Median: **11.0**

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

## STAR Dataset Composition (Reference Only)
- Total STAR questions in source file: **1990**
- By category:
  - `Interaction`: **500**
  - `Prediction`: **500**
  - `Sequence`: **500**
  - `Feasibility`: **490**
- By question type/template:
  - `T2`: **402**
  - `T4`: **366**
  - `T1`: **354**
  - `T3`: **336**
  - `T5`: **309**
  - `T6`: **223**

## Notes
- All performance stats in this report are run-based (each JSON file is one run).
- Correctness trajectories are computed from per-turn answer letters against the run gold answer.
- `C` means correct, `W` means wrong; condensed trajectories collapse repeated same-state turns.
- `first_flip_turn` is the first turn index where answer letter differs from the previous turn.
