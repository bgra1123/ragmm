
## Stage 1 = SGD: two retrains disagree (lower is better); cell = mean over 200 seeds

**german** (stage 1: 25.1%, changed 55.2)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 25.3% | 15.0% | 14.1% | 13.8% | 23.2% | 12.7% | C3 |
| 2 | 25.5% | 12.9% | 10.8% | 8.2% | 21.3% | 8.4% | C3 |
| 5 | 25.4% | 9.7% | 7.1% | 3.1% | 15.8% | 3.5% | C3 |
| 20 | 25.8% | 6.8% | 3.8% | 0.1% | 2.3% | 0.4% | C3 |

**compas** (stage 1: 29.2%, changed 398.8)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 29.4% | 9.0% | 3.4% | 10.7% | 26.0% | 11.1% | C3 |
| 2 | 28.3% | 7.5% | 2.4% | 6.3% | 22.9% | 7.2% | C3 |
| 5 | 28.7% | 5.7% | 1.3% | 1.4% | 14.8% | 1.4% | C3 |
| 20 | 28.4% | 4.2% | 0.5% | 0.0% | 1.0% | 0.0% | C3 |

**pima** (stage 1: 14.0%, changed 22.5)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 14.2% | 6.5% | 5.5% | 5.1% | 12.2% | 4.8% | C3 |
| 2 | 14.7% | 5.2% | 3.9% | 2.0% | 10.4% | 1.9% | C3 |
| 5 | 14.3% | 4.0% | 2.3% | 0.2% | 6.6% | 0.2% | C3 |
| 20 | 14.7% | 2.8% | 1.1% | 0.0% | 0.4% | 0.0% | C3 |

**heart** (stage 1: 7.6%, changed 4.8)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 7.8% | 5.3% | 5.3% | 3.3% | 6.2% | 2.8% | C2 |
| 2 | 7.9% | 4.6% | 4.4% | 1.9% | 5.1% | 1.8% | C3 |
| 5 | 7.8% | 3.7% | 3.0% | 0.6% | 3.1% | 0.6% | C3 |
| 20 | 7.9% | 2.5% | 2.0% | 0.0% | 0.1% | 0.0% | C3 |


## Stage 1 = SGD: risk arms vs best continuation (difference in disagreement, 95% CI; negative = risk arm better)

| Dataset | Extra epochs | Arm | disagreement | changed decisions / 300-ish (mean) | wrong-decision rate |
|---|---|---|---|---|---|
| german | 1 | first-order (theta=1) | -0.3 [-0.8, +0.2] | -0.5 [-1.7, +0.8] | +0.1 [-0.2, +0.4] |
| german | 1 | exact-change (theta=5) | +9.1 [+8.5, +9.6] | +20.3 [+18.8, +21.8] | +2.3 [+2.0, +2.6] |
| german | 2 | first-order (theta=1) | -2.7 [-3.0, -2.3] | -5.7 [-6.7, -4.6] | -0.2 [-0.4, +0.1] |
| german | 2 | exact-change (theta=5) | +10.5 [+9.9, +10.9] | +22.5 [+21.2, +24.0] | +2.2 [+1.9, +2.6] |
| german | 5 | first-order (theta=1) | -4.0 [-4.2, -3.7] | -8.8 [-9.5, -8.1] | -0.4 [-0.6, -0.2] |
| german | 5 | exact-change (theta=5) | +8.7 [+8.2, +9.1] | +18.1 [+16.9, +19.2] | +1.2 [+0.9, +1.5] |
| german | 20 | first-order (theta=1) | -3.6 [-3.8, -3.5] | -9.5 [-9.9, -9.2] | -0.5 [-0.6, -0.4] |
| german | 20 | exact-change (theta=5) | -1.4 [-1.6, -1.3] | -4.1 [-4.6, -3.6] | -0.3 [-0.5, -0.2] |
| compas | 1 | first-order (theta=1) | +7.3 [+6.7, +7.8] | +87.4 [+79.2, +95.8] | +0.4 [+0.2, +0.5] |
| compas | 1 | exact-change (theta=5) | +22.6 [+21.0, +24.1] | +297.3 [+275.3, +320.1] | +3.2 [+2.8, +3.7] |
| compas | 2 | first-order (theta=1) | +3.9 [+3.4, +4.4] | +40.7 [+34.4, +47.5] | +0.0 [-0.0, +0.1] |
| compas | 2 | exact-change (theta=5) | +20.6 [+19.1, +21.8] | +263.4 [+244.2, +283.6] | +2.5 [+2.2, +2.9] |
| compas | 5 | first-order (theta=1) | +0.2 [-0.1, +0.4] | -12.6 [-15.4, -9.7] | -0.1 [-0.1, -0.1] |
| compas | 5 | exact-change (theta=5) | +13.6 [+12.5, +14.5] | +161.6 [+148.8, +174.7] | +1.0 [+0.8, +1.2] |
| compas | 20 | first-order (theta=1) | -0.5 [-0.5, -0.4] | -25.6 [-26.2, -25.0] | -0.1 [-0.1, -0.1] |
| compas | 20 | exact-change (theta=5) | +0.5 [+0.4, +0.6] | -12.8 [-13.8, -11.8] | -0.0 [-0.1, -0.0] |
| pima | 1 | first-order (theta=1) | -0.4 [-0.8, +0.0] | -0.7 [-1.5, +0.0] | +0.2 [-0.0, +0.4] |
| pima | 1 | exact-change (theta=5) | +6.7 [+5.9, +7.4] | +10.8 [+9.6, +12.2] | +1.7 [+1.3, +2.0] |
| pima | 2 | first-order (theta=1) | -1.9 [-2.2, -1.7] | -3.3 [-3.9, -2.8] | +0.2 [+0.0, +0.4] |
| pima | 2 | exact-change (theta=5) | +6.5 [+5.8, +7.1] | +10.3 [+9.2, +11.4] | +1.4 [+1.1, +1.7] |
| pima | 5 | first-order (theta=1) | -2.1 [-2.2, -2.0] | -3.6 [-3.9, -3.3] | +0.5 [+0.4, +0.6] |
| pima | 5 | exact-change (theta=5) | +4.3 [+3.9, +4.7] | +6.3 [+5.6, +7.1] | +0.8 [+0.6, +1.0] |
| pima | 20 | first-order (theta=1) | -1.1 [-1.2, -1.0] | -2.4 [-2.6, -2.2] | +0.7 [+0.6, +0.7] |
| pima | 20 | exact-change (theta=5) | -0.7 [-0.8, -0.6] | -1.8 [-2.1, -1.6] | +0.5 [+0.4, +0.6] |
| heart | 1 | first-order (theta=1) | -2.0 [-2.3, -1.7] | -0.9 [-1.2, -0.7] | -0.2 [-0.5, +0.0] |
| heart | 1 | exact-change (theta=5) | +0.9 [+0.5, +1.2] | +0.5 [+0.3, +0.8] | +0.0 [-0.3, +0.3] |
| heart | 2 | first-order (theta=1) | -2.5 [-2.7, -2.2] | -1.6 [-1.8, -1.4] | -0.3 [-0.5, -0.0] |
| heart | 2 | exact-change (theta=5) | +0.7 [+0.3, +1.1] | +0.4 [+0.1, +0.7] | +0.2 [-0.1, +0.5] |
| heart | 5 | first-order (theta=1) | -2.3 [-2.6, -2.1] | -1.8 [-1.9, -1.6] | -0.5 [-0.7, -0.4] |
| heart | 5 | exact-change (theta=5) | +0.1 [-0.2, +0.4] | -0.1 [-0.4, +0.1] | +0.0 [-0.2, +0.2] |
| heart | 20 | first-order (theta=1) | -2.0 [-2.1, -1.9] | -1.4 [-1.6, -1.3] | -0.7 [-0.8, -0.6] |
| heart | 20 | exact-change (theta=5) | -2.0 [-2.1, -1.9] | -1.4 [-1.5, -1.3] | -0.6 [-0.8, -0.5] |

## Stage 1 = SGD: wrong-decision rate and per-row harm

| Dataset | Extra epochs | arm | wrong decision | share rows worse | p95 row loss increase | mean step x L | seconds (200 seeds) |
|---|---|---|---|---|---|---|---|
| german | 1 | C1 continue | 31.0% | 0.50 | +1.902 | nan | 0.0 |
| german | 1 | C2 continue, decayed | 28.3% | 0.45 | +0.851 | nan | 0.0 |
| german | 1 | C3 continue, averaged | 28.2% | 0.42 | +0.801 | nan | 0.0 |
| german | 1 | R risk first-order (theta=1) | 28.3% | 0.40 | +0.503 | 3.52 | 0.6 |
| german | 1 | R risk exact-change (theta=5) | 30.5% | 0.34 | +0.056 | 0.60 | 2.9 |
| german | 1 | L exact line search | 28.3% | 0.41 | +0.695 | 4.29 | 0.9 |
| german | 2 | C1 continue | 30.9% | 0.50 | +1.889 | nan | 0.0 |
| german | 2 | C2 continue, decayed | 28.1% | 0.46 | +0.757 | nan | 0.0 |
| german | 2 | C3 continue, averaged | 27.8% | 0.41 | +0.746 | nan | 0.0 |
| german | 2 | R risk first-order (theta=1) | 27.7% | 0.43 | +0.511 | 3.98 | 0.8 |
| german | 2 | R risk exact-change (theta=5) | 30.1% | 0.35 | +0.111 | 0.61 | 5.1 |
| german | 2 | L exact line search | 27.6% | 0.42 | +0.459 | 3.84 | 1.3 |
| german | 5 | C1 continue | 30.8% | 0.50 | +1.908 | nan | 0.1 |
| german | 5 | C2 continue, decayed | 27.6% | 0.48 | +0.641 | nan | 0.1 |
| german | 5 | C3 continue, averaged | 27.5% | 0.41 | +0.683 | nan | 0.1 |
| german | 5 | R risk first-order (theta=1) | 27.1% | 0.48 | +0.533 | 3.55 | 0.8 |
| german | 5 | R risk exact-change (theta=5) | 28.7% | 0.38 | +0.261 | 0.68 | 4.7 |
| german | 5 | L exact line search | 27.1% | 0.47 | +0.539 | 3.35 | 2.8 |
| german | 20 | C1 continue | 30.8% | 0.50 | +1.919 | nan | 0.4 |
| german | 20 | C2 continue, decayed | 27.2% | 0.50 | +0.576 | nan | 0.4 |
| german | 20 | C3 continue, averaged | 27.2% | 0.40 | +0.641 | nan | 0.3 |
| german | 20 | R risk first-order (theta=1) | 26.7% | 0.52 | +0.523 | 4.51 | 6.4 |
| german | 20 | R risk exact-change (theta=5) | 26.9% | 0.50 | +0.517 | 1.26 | 22.0 |
| german | 20 | L exact line search | 26.8% | 0.52 | +0.522 | 2.93 | 9.7 |
| compas | 1 | C1 continue | 37.3% | 0.50 | +0.765 | nan | 0.0 |
| compas | 1 | C2 continue, decayed | 33.4% | 0.52 | +0.370 | nan | 0.0 |
| compas | 1 | C3 continue, averaged | 33.1% | 0.49 | +0.383 | nan | 0.0 |
| compas | 1 | R risk first-order (theta=1) | 33.5% | 0.50 | +0.378 | 2.05 | 3.9 |
| compas | 1 | R risk exact-change (theta=5) | 36.4% | 0.49 | +0.051 | 0.34 | 10.1 |
| compas | 1 | L exact line search | 33.5% | 0.50 | +0.404 | 2.17 | 4.0 |
| compas | 2 | C1 continue | 36.9% | 0.50 | +0.705 | nan | 0.1 |
| compas | 2 | C2 continue, decayed | 33.2% | 0.51 | +0.364 | nan | 0.1 |
| compas | 2 | C3 continue, averaged | 33.1% | 0.49 | +0.378 | nan | 0.1 |
| compas | 2 | R risk first-order (theta=1) | 33.2% | 0.51 | +0.309 | 2.29 | 6.0 |
| compas | 2 | R risk exact-change (theta=5) | 35.6% | 0.49 | +0.097 | 0.35 | 17.9 |
| compas | 2 | L exact line search | 33.2% | 0.51 | +0.301 | 2.33 | 8.2 |
| compas | 5 | C1 continue | 37.1% | 0.49 | +0.733 | nan | 0.3 |
| compas | 5 | C2 continue, decayed | 33.2% | 0.52 | +0.359 | nan | 0.3 |
| compas | 5 | C3 continue, averaged | 33.2% | 0.49 | +0.379 | nan | 0.4 |
| compas | 5 | R risk first-order (theta=1) | 33.1% | 0.52 | +0.355 | 2.13 | 11.9 |
| compas | 5 | R risk exact-change (theta=5) | 34.2% | 0.50 | +0.205 | 0.38 | 42.8 |
| compas | 5 | L exact line search | 33.1% | 0.52 | +0.355 | 2.12 | 25.2 |
| compas | 20 | C1 continue | 36.9% | 0.50 | +0.705 | nan | 0.5 |
| compas | 20 | C2 continue, decayed | 33.1% | 0.52 | +0.355 | nan | 0.8 |
| compas | 20 | C3 continue, averaged | 33.2% | 0.49 | +0.380 | nan | 0.7 |
| compas | 20 | R risk first-order (theta=1) | 33.1% | 0.52 | +0.352 | 2.10 | 40.4 |
| compas | 20 | R risk exact-change (theta=5) | 33.2% | 0.52 | +0.344 | 0.57 | 213.8 |
| compas | 20 | L exact line search | 33.1% | 0.52 | +0.352 | 2.03 | 115.4 |
| pima | 1 | C1 continue | 25.7% | 0.51 | +0.776 | nan | 0.0 |
| pima | 1 | C2 continue, decayed | 23.9% | 0.51 | +0.450 | nan | 0.0 |
| pima | 1 | C3 continue, averaged | 23.8% | 0.49 | +0.412 | nan | 0.0 |
| pima | 1 | R risk first-order (theta=1) | 24.0% | 0.49 | +0.364 | 2.58 | 0.3 |
| pima | 1 | R risk exact-change (theta=5) | 25.4% | 0.48 | +0.047 | 0.43 | 2.3 |
| pima | 1 | L exact line search | 23.8% | 0.49 | +0.381 | 2.65 | 1.2 |
| pima | 2 | C1 continue | 25.9% | 0.51 | +0.763 | nan | 0.0 |
| pima | 2 | C2 continue, decayed | 23.8% | 0.53 | +0.423 | nan | 0.0 |
| pima | 2 | C3 continue, averaged | 23.5% | 0.49 | +0.401 | nan | 0.0 |
| pima | 2 | R risk first-order (theta=1) | 23.7% | 0.53 | +0.328 | 3.00 | 0.2 |
| pima | 2 | R risk exact-change (theta=5) | 24.9% | 0.48 | +0.090 | 0.44 | 2.3 |
| pima | 2 | L exact line search | 23.7% | 0.53 | +0.304 | 2.89 | 1.8 |
| pima | 5 | C1 continue | 25.7% | 0.51 | +0.760 | nan | 0.0 |
| pima | 5 | C2 continue, decayed | 23.6% | 0.53 | +0.396 | nan | 0.0 |
| pima | 5 | C3 continue, averaged | 23.3% | 0.49 | +0.390 | nan | 0.0 |
| pima | 5 | R risk first-order (theta=1) | 23.8% | 0.54 | +0.352 | 3.08 | 0.7 |
| pima | 5 | R risk exact-change (theta=5) | 24.1% | 0.50 | +0.193 | 0.47 | 3.2 |
| pima | 5 | L exact line search | 23.8% | 0.54 | +0.354 | 2.79 | 1.8 |
| pima | 20 | C1 continue | 26.0% | 0.51 | +0.767 | nan | 0.1 |
| pima | 20 | C2 continue, decayed | 23.5% | 0.54 | +0.371 | nan | 0.1 |
| pima | 20 | C3 continue, averaged | 23.3% | 0.48 | +0.386 | nan | 0.1 |
| pima | 20 | R risk first-order (theta=1) | 23.9% | 0.55 | +0.353 | 3.29 | 2.9 |
| pima | 20 | R risk exact-change (theta=5) | 23.8% | 0.54 | +0.346 | 0.75 | 20.3 |
| pima | 20 | L exact line search | 23.9% | 0.55 | +0.353 | 2.81 | 7.7 |
| heart | 1 | C1 continue | 21.6% | 0.51 | +0.452 | nan | 0.0 |
| heart | 1 | C2 continue, decayed | 21.1% | 0.47 | +0.282 | nan | 0.0 |
| heart | 1 | C3 continue, averaged | 21.1% | 0.47 | +0.257 | nan | 0.0 |
| heart | 1 | R risk first-order (theta=1) | 20.9% | 0.44 | +0.335 | 7.03 | 0.2 |
| heart | 1 | R risk exact-change (theta=5) | 21.1% | 0.39 | +0.038 | 1.11 | 0.6 |
| heart | 1 | L exact line search | 20.7% | 0.43 | +0.251 | 5.59 | 0.3 |
| heart | 2 | C1 continue | 21.5% | 0.51 | +0.489 | nan | 0.0 |
| heart | 2 | C2 continue, decayed | 21.0% | 0.47 | +0.297 | nan | 0.0 |
| heart | 2 | C3 continue, averaged | 20.8% | 0.46 | +0.263 | nan | 0.0 |
| heart | 2 | R risk first-order (theta=1) | 20.5% | 0.44 | +0.198 | 6.48 | 0.6 |
| heart | 2 | R risk exact-change (theta=5) | 21.0% | 0.40 | +0.072 | 1.15 | 1.1 |
| heart | 2 | L exact line search | 20.5% | 0.44 | +0.199 | 5.75 | 0.3 |
| heart | 5 | C1 continue | 21.6% | 0.50 | +0.479 | nan | 0.0 |
| heart | 5 | C2 continue, decayed | 20.8% | 0.46 | +0.288 | nan | 0.0 |
| heart | 5 | C3 continue, averaged | 20.7% | 0.44 | +0.253 | nan | 0.0 |
| heart | 5 | R risk first-order (theta=1) | 20.1% | 0.45 | +0.235 | 5.94 | 0.6 |
| heart | 5 | R risk exact-change (theta=5) | 20.7% | 0.42 | +0.149 | 1.30 | 2.1 |
| heart | 5 | L exact line search | 20.1% | 0.45 | +0.233 | 5.65 | 1.6 |
| heart | 20 | C1 continue | 21.6% | 0.51 | +0.491 | nan | 0.1 |
| heart | 20 | C2 continue, decayed | 20.7% | 0.46 | +0.257 | nan | 0.1 |
| heart | 20 | C3 continue, averaged | 20.5% | 0.43 | +0.240 | nan | 0.1 |
| heart | 20 | R risk first-order (theta=1) | 19.8% | 0.46 | +0.234 | 6.01 | 0.8 |
| heart | 20 | R risk exact-change (theta=5) | 19.8% | 0.46 | +0.233 | 2.83 | 4.6 |
| heart | 20 | L exact line search | 19.8% | 0.46 | +0.234 | 5.81 | 2.5 |

## Stage 1 = Adam: two retrains disagree (lower is better); cell = mean over 200 seeds

**german** (stage 1: 8.4%, changed 18.1)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 9.0% | 8.5% | 8.4% | 3.7% | 7.4% | 3.7% | C3 |
| 2 | 8.7% | 7.4% | 7.2% | 2.5% | 6.5% | 2.5% | C3 |
| 5 | 8.4% | 5.6% | 5.2% | 1.3% | 4.2% | 1.3% | C3 |
| 20 | 8.4% | 3.9% | 3.0% | 0.0% | 0.8% | 0.1% | C3 |

**compas** (stage 1: 7.2%, changed 91.1)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 7.2% | 4.0% | 3.4% | 1.9% | 5.9% | 1.8% | C3 |
| 2 | 6.9% | 3.6% | 2.4% | 0.8% | 4.8% | 0.8% | C3 |
| 5 | 7.3% | 2.3% | 1.3% | 0.2% | 2.5% | 0.2% | C3 |
| 20 | 7.1% | 1.6% | 0.6% | 0.0% | 0.1% | 0.0% | C3 |

**pima** (stage 1: 2.6%, changed 3.9)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 2.9% | 2.6% | 2.6% | 0.7% | 2.1% | 0.7% | C3 |
| 2 | 2.7% | 2.4% | 2.3% | 0.3% | 1.7% | 0.3% | C3 |
| 5 | 2.4% | 1.6% | 1.5% | 0.2% | 0.9% | 0.2% | C3 |
| 20 | 2.8% | 1.0% | 0.7% | 0.0% | 0.2% | 0.0% | C3 |

**heart** (stage 1: 2.5%, changed 1.7)

| Extra epochs | C1 cont. | C2 cont., decayed | C3 cont., averaged | R first-order (theta=1) | R exact-change (theta=5) | L exact line search | best continuation |
|---|---|---|---|---|---|---|---|
| 1 | 2.6% | 2.4% | 2.4% | 1.5% | 2.3% | 1.3% | C3 |
| 2 | 2.7% | 2.5% | 2.4% | 0.7% | 2.0% | 0.6% | C3 |
| 5 | 2.5% | 2.3% | 2.3% | 0.0% | 1.3% | 0.0% | C3 |
| 20 | 2.6% | 1.9% | 1.8% | 0.0% | 0.0% | 0.0% | C3 |


## Stage 1 = Adam: risk arms vs best continuation (difference in disagreement, 95% CI; negative = risk arm better)

| Dataset | Extra epochs | Arm | disagreement | changed decisions / 300-ish (mean) | wrong-decision rate |
|---|---|---|---|---|---|
| german | 1 | first-order (theta=1) | -4.7 [-5.0, -4.5] | -9.9 [-10.6, -9.3] | -0.5 [-0.7, -0.3] |
| german | 1 | exact-change (theta=5) | -1.0 [-1.3, -0.8] | -2.2 [-2.9, -1.6] | -0.2 [-0.4, +0.0] |
| german | 2 | first-order (theta=1) | -4.7 [-5.0, -4.5] | -9.9 [-10.4, -9.3] | -0.6 [-0.8, -0.4] |
| german | 2 | exact-change (theta=5) | -0.7 [-1.0, -0.4] | -1.6 [-2.3, -0.8] | -0.1 [-0.3, +0.1] |
| german | 5 | first-order (theta=1) | -3.9 [-4.1, -3.7] | -8.5 [-9.0, -8.0] | -0.6 [-0.8, -0.5] |
| german | 5 | exact-change (theta=5) | -1.0 [-1.2, -0.7] | -2.2 [-2.7, -1.6] | -0.2 [-0.4, -0.0] |
| german | 20 | first-order (theta=1) | -3.0 [-3.1, -2.9] | -6.8 [-7.1, -6.4] | -0.3 [-0.4, -0.1] |
| german | 20 | exact-change (theta=5) | -2.2 [-2.3, -2.1] | -5.2 [-5.5, -4.8] | -0.2 [-0.4, -0.1] |
| compas | 1 | first-order (theta=1) | -1.5 [-1.7, -1.2] | -18.3 [-21.6, -14.9] | -0.0 [-0.1, +0.0] |
| compas | 1 | exact-change (theta=5) | +2.5 [+2.0, +3.0] | +31.4 [+24.8, +37.7] | +0.1 [-0.0, +0.1] |
| compas | 2 | first-order (theta=1) | -1.6 [-1.8, -1.4] | -19.2 [-21.5, -17.0] | -0.1 [-0.1, -0.0] |
| compas | 2 | exact-change (theta=5) | +2.4 [+2.0, +2.8] | +29.6 [+24.4, +34.8] | +0.0 [-0.0, +0.1] |
| compas | 5 | first-order (theta=1) | -1.1 [-1.3, -1.0] | -15.2 [-16.7, -13.9] | -0.0 [-0.0, +0.0] |
| compas | 5 | exact-change (theta=5) | +1.2 [+0.9, +1.4] | +13.8 [+10.8, +16.8] | +0.1 [+0.0, +0.1] |
| compas | 20 | first-order (theta=1) | -0.6 [-0.7, -0.6] | -9.5 [-10.1, -9.0] | +0.0 [+0.0, +0.1] |
| compas | 20 | exact-change (theta=5) | -0.5 [-0.6, -0.5] | -8.3 [-8.9, -7.7] | +0.1 [+0.0, +0.1] |
| pima | 1 | first-order (theta=1) | -1.9 [-2.0, -1.7] | -2.9 [-3.2, -2.6] | +0.2 [+0.0, +0.3] |
| pima | 1 | exact-change (theta=5) | -0.5 [-0.7, -0.3] | -0.9 [-1.2, -0.5] | +0.0 [-0.1, +0.1] |
| pima | 2 | first-order (theta=1) | -2.1 [-2.2, -1.9] | -3.1 [-3.4, -2.8] | +0.3 [+0.2, +0.4] |
| pima | 2 | exact-change (theta=5) | -0.7 [-0.9, -0.5] | -1.1 [-1.5, -0.8] | +0.1 [-0.0, +0.2] |
| pima | 5 | first-order (theta=1) | -1.3 [-1.4, -1.2] | -2.1 [-2.3, -1.9] | +0.4 [+0.3, +0.4] |
| pima | 5 | exact-change (theta=5) | -0.6 [-0.7, -0.5] | -1.0 [-1.2, -0.8] | +0.2 [+0.1, +0.3] |
| pima | 20 | first-order (theta=1) | -0.7 [-0.8, -0.7] | -1.3 [-1.4, -1.1] | +0.4 [+0.3, +0.4] |
| pima | 20 | exact-change (theta=5) | -0.5 [-0.6, -0.4] | -0.9 [-1.0, -0.7] | +0.2 [+0.1, +0.3] |
| heart | 1 | first-order (theta=1) | -0.9 [-1.1, -0.7] | -0.7 [-0.9, -0.5] | -0.1 [-0.3, +0.0] |
| heart | 1 | exact-change (theta=5) | -0.2 [-0.3, -0.0] | -0.1 [-0.2, +0.1] | -0.0 [-0.1, +0.1] |
| heart | 2 | first-order (theta=1) | -1.8 [-2.0, -1.6] | -1.3 [-1.4, -1.1] | -0.5 [-0.6, -0.3] |
| heart | 2 | exact-change (theta=5) | -0.5 [-0.6, -0.3] | -0.3 [-0.5, -0.2] | -0.1 [-0.2, +0.0] |
| heart | 5 | first-order (theta=1) | -2.3 [-2.4, -2.2] | -1.6 [-1.7, -1.4] | -0.9 [-1.0, -0.7] |
| heart | 5 | exact-change (theta=5) | -1.0 [-1.2, -0.8] | -0.8 [-1.0, -0.7] | -0.5 [-0.7, -0.4] |
| heart | 20 | first-order (theta=1) | -1.8 [-1.9, -1.6] | -1.2 [-1.3, -1.0] | -0.6 [-0.7, -0.5] |
| heart | 20 | exact-change (theta=5) | -1.8 [-1.9, -1.6] | -1.2 [-1.3, -1.0] | -0.6 [-0.7, -0.5] |

## Stage 1 = Adam: wrong-decision rate and per-row harm

| Dataset | Extra epochs | arm | wrong decision | share rows worse | p95 row loss increase | mean step x L | seconds (200 seeds) |
|---|---|---|---|---|---|---|---|
| german | 1 | C1 continue | 27.6% | 0.50 | +0.329 | nan | 0.0 |
| german | 1 | C2 continue, decayed | 27.5% | 0.50 | +0.245 | nan | 0.0 |
| german | 1 | C3 continue, averaged | 27.5% | 0.50 | +0.233 | nan | 0.0 |
| german | 1 | R risk first-order (theta=1) | 27.0% | 0.50 | +0.202 | 3.12 | 1.9 |
| german | 1 | R risk exact-change (theta=5) | 27.3% | 0.50 | +0.030 | 0.52 | 3.0 |
| german | 1 | L exact line search | 27.0% | 0.50 | +0.201 | 3.10 | 1.2 |
| german | 2 | C1 continue | 27.5% | 0.50 | +0.344 | nan | 0.0 |
| german | 2 | C2 continue, decayed | 27.3% | 0.50 | +0.261 | nan | 0.0 |
| german | 2 | C3 continue, averaged | 27.3% | 0.50 | +0.239 | nan | 0.0 |
| german | 2 | R risk first-order (theta=1) | 26.7% | 0.51 | +0.175 | 2.94 | 0.7 |
| german | 2 | R risk exact-change (theta=5) | 27.2% | 0.50 | +0.056 | 0.54 | 2.5 |
| german | 2 | L exact line search | 26.7% | 0.51 | +0.174 | 2.87 | 1.0 |
| german | 5 | C1 continue | 27.5% | 0.50 | +0.351 | nan | 0.1 |
| german | 5 | C2 continue, decayed | 27.2% | 0.50 | +0.260 | nan | 0.1 |
| german | 5 | C3 continue, averaged | 27.2% | 0.49 | +0.231 | nan | 0.1 |
| german | 5 | R risk first-order (theta=1) | 26.5% | 0.52 | +0.206 | 2.87 | 2.0 |
| german | 5 | R risk exact-change (theta=5) | 26.9% | 0.50 | +0.118 | 0.60 | 4.7 |
| german | 5 | L exact line search | 26.5% | 0.52 | +0.205 | 2.76 | 3.0 |
| german | 20 | C1 continue | 27.5% | 0.50 | +0.357 | nan | 0.6 |
| german | 20 | C2 continue, decayed | 27.0% | 0.51 | +0.241 | nan | 0.6 |
| german | 20 | C3 continue, averaged | 26.9% | 0.50 | +0.220 | nan | 0.6 |
| german | 20 | R risk first-order (theta=1) | 26.7% | 0.53 | +0.208 | 4.78 | 5.7 |
| german | 20 | R risk exact-change (theta=5) | 26.7% | 0.52 | +0.205 | 1.10 | 21.4 |
| german | 20 | L exact line search | 26.7% | 0.53 | +0.208 | 2.77 | 12.1 |
| compas | 1 | C1 continue | 33.2% | 0.50 | +0.110 | nan | 0.0 |
| compas | 1 | C2 continue, decayed | 33.2% | 0.50 | +0.088 | nan | 0.0 |
| compas | 1 | C3 continue, averaged | 33.1% | 0.49 | +0.079 | nan | 0.0 |
| compas | 1 | R risk first-order (theta=1) | 33.1% | 0.50 | +0.084 | 2.04 | 10.3 |
| compas | 1 | R risk exact-change (theta=5) | 33.2% | 0.50 | +0.013 | 0.34 | 13.4 |
| compas | 1 | L exact line search | 33.1% | 0.50 | +0.082 | 2.01 | 3.5 |
| compas | 2 | C1 continue | 33.2% | 0.50 | +0.117 | nan | 0.1 |
| compas | 2 | C2 continue, decayed | 33.1% | 0.50 | +0.092 | nan | 0.1 |
| compas | 2 | C3 continue, averaged | 33.1% | 0.49 | +0.081 | nan | 0.1 |
| compas | 2 | R risk first-order (theta=1) | 33.1% | 0.50 | +0.069 | 1.97 | 11.6 |
| compas | 2 | R risk exact-change (theta=5) | 33.2% | 0.50 | +0.025 | 0.35 | 25.0 |
| compas | 2 | L exact line search | 33.1% | 0.50 | +0.070 | 1.98 | 8.6 |
| compas | 5 | C1 continue | 33.2% | 0.49 | +0.122 | nan | 0.2 |
| compas | 5 | C2 continue, decayed | 33.2% | 0.50 | +0.085 | nan | 0.2 |
| compas | 5 | C3 continue, averaged | 33.1% | 0.50 | +0.080 | nan | 0.3 |
| compas | 5 | R risk first-order (theta=1) | 33.1% | 0.50 | +0.080 | 1.96 | 21.1 |
| compas | 5 | R risk exact-change (theta=5) | 33.2% | 0.50 | +0.049 | 0.38 | 56.8 |
| compas | 5 | L exact line search | 33.1% | 0.50 | +0.080 | 1.97 | 26.8 |
| compas | 20 | C1 continue | 33.2% | 0.50 | +0.118 | nan | 0.7 |
| compas | 20 | C2 continue, decayed | 33.1% | 0.50 | +0.082 | nan | 0.7 |
| compas | 20 | C3 continue, averaged | 33.1% | 0.50 | +0.080 | nan | 0.7 |
| compas | 20 | R risk first-order (theta=1) | 33.1% | 0.50 | +0.080 | 2.09 | 41.8 |
| compas | 20 | R risk exact-change (theta=5) | 33.1% | 0.50 | +0.078 | 0.55 | 220.1 |
| compas | 20 | L exact line search | 33.1% | 0.50 | +0.080 | 2.01 | 114.7 |
| pima | 1 | C1 continue | 23.4% | 0.48 | +0.121 | nan | 0.0 |
| pima | 1 | C2 continue, decayed | 23.4% | 0.48 | +0.084 | nan | 0.0 |
| pima | 1 | C3 continue, averaged | 23.4% | 0.48 | +0.082 | nan | 0.0 |
| pima | 1 | R risk first-order (theta=1) | 23.6% | 0.50 | +0.086 | 2.31 | 0.1 |
| pima | 1 | R risk exact-change (theta=5) | 23.5% | 0.50 | +0.014 | 0.39 | 0.6 |
| pima | 1 | L exact line search | 23.6% | 0.50 | +0.090 | 2.37 | 0.4 |
| pima | 2 | C1 continue | 23.5% | 0.49 | +0.134 | nan | 0.0 |
| pima | 2 | C2 continue, decayed | 23.4% | 0.48 | +0.101 | nan | 0.0 |
| pima | 2 | C3 continue, averaged | 23.4% | 0.48 | +0.095 | nan | 0.0 |
| pima | 2 | R risk first-order (theta=1) | 23.7% | 0.51 | +0.084 | 2.72 | 0.3 |
| pima | 2 | R risk exact-change (theta=5) | 23.5% | 0.50 | +0.026 | 0.40 | 1.3 |
| pima | 2 | L exact line search | 23.7% | 0.51 | +0.080 | 2.62 | 0.8 |
| pima | 5 | C1 continue | 23.3% | 0.49 | +0.140 | nan | 0.0 |
| pima | 5 | C2 continue, decayed | 23.4% | 0.48 | +0.109 | nan | 0.0 |
| pima | 5 | C3 continue, averaged | 23.4% | 0.48 | +0.098 | nan | 0.0 |
| pima | 5 | R risk first-order (theta=1) | 23.8% | 0.51 | +0.092 | 2.96 | 0.9 |
| pima | 5 | R risk exact-change (theta=5) | 23.6% | 0.50 | +0.053 | 0.44 | 2.9 |
| pima | 5 | L exact line search | 23.7% | 0.51 | +0.093 | 2.66 | 1.9 |
| pima | 20 | C1 continue | 23.4% | 0.50 | +0.146 | nan | 0.1 |
| pima | 20 | C2 continue, decayed | 23.5% | 0.50 | +0.106 | nan | 0.1 |
| pima | 20 | C3 continue, averaged | 23.5% | 0.49 | +0.097 | nan | 0.1 |
| pima | 20 | R risk first-order (theta=1) | 23.9% | 0.51 | +0.093 | 3.22 | 8.6 |
| pima | 20 | R risk exact-change (theta=5) | 23.7% | 0.51 | +0.091 | 0.72 | 15.6 |
| pima | 20 | L exact line search | 23.9% | 0.51 | +0.093 | 2.78 | 9.0 |
| heart | 1 | C1 continue | 20.5% | 0.51 | +0.115 | nan | 0.0 |
| heart | 1 | C2 continue, decayed | 20.4% | 0.50 | +0.079 | nan | 0.0 |
| heart | 1 | C3 continue, averaged | 20.4% | 0.50 | +0.079 | nan | 0.0 |
| heart | 1 | R risk first-order (theta=1) | 20.3% | 0.49 | +0.133 | 6.69 | 0.1 |
| heart | 1 | R risk exact-change (theta=5) | 20.4% | 0.47 | +0.019 | 1.06 | 0.4 |
| heart | 1 | L exact line search | 20.3% | 0.48 | +0.101 | 5.18 | 0.1 |
| heart | 2 | C1 continue | 20.7% | 0.51 | +0.138 | nan | 0.0 |
| heart | 2 | C2 continue, decayed | 20.5% | 0.51 | +0.100 | nan | 0.0 |
| heart | 2 | C3 continue, averaged | 20.5% | 0.50 | +0.098 | nan | 0.0 |
| heart | 2 | R risk first-order (theta=1) | 20.0% | 0.49 | +0.084 | 6.02 | 0.1 |
| heart | 2 | R risk exact-change (theta=5) | 20.4% | 0.47 | +0.035 | 1.10 | 0.4 |
| heart | 2 | L exact line search | 20.0% | 0.49 | +0.085 | 5.37 | 0.2 |
| heart | 5 | C1 continue | 20.6% | 0.51 | +0.152 | nan | 0.0 |
| heart | 5 | C2 continue, decayed | 20.7% | 0.51 | +0.118 | nan | 0.0 |
| heart | 5 | C3 continue, averaged | 20.7% | 0.50 | +0.110 | nan | 0.0 |
| heart | 5 | R risk first-order (theta=1) | 19.8% | 0.49 | +0.100 | 5.71 | 0.3 |
| heart | 5 | R risk exact-change (theta=5) | 20.1% | 0.48 | +0.068 | 1.26 | 1.2 |
| heart | 5 | L exact line search | 19.8% | 0.49 | +0.099 | 5.47 | 0.7 |
| heart | 20 | C1 continue | 20.6% | 0.50 | +0.156 | nan | 0.1 |
| heart | 20 | C2 continue, decayed | 20.5% | 0.48 | +0.119 | nan | 0.1 |
| heart | 20 | C3 continue, averaged | 20.4% | 0.48 | +0.106 | nan | 0.1 |
| heart | 20 | R risk first-order (theta=1) | 19.8% | 0.49 | +0.100 | 6.03 | 0.9 |
| heart | 20 | R risk exact-change (theta=5) | 19.8% | 0.49 | +0.099 | 3.89 | 5.5 |
| heart | 20 | L exact line search | 19.8% | 0.49 | +0.100 | 5.87 | 3.0 |

## German Credit, worst group wrong-decision rate (groups with >= 30 held-out rows)


Stage 1 = SGD

| Extra epochs | C1 continue | C2 continue, decayed | C3 continue, averaged | R risk first-order (theta=1) | R risk exact-change (theta=5) | L exact line search |
|---|---|---|---|---|---|---|
| 1 | 41.8% | 39.2% | 39.3% | 39.0% | 41.0% | 39.2% |
| 2 | 41.5% | 39.1% | 38.8% | 39.3% | 40.6% | 39.2% |
| 5 | 41.1% | 39.0% | 38.7% | 39.1% | 39.5% | 38.9% |
| 20 | 41.2% | 38.7% | 38.3% | 39.0% | 38.9% | 39.0% |

Stage 1 = Adam

| Extra epochs | C1 continue | C2 continue, decayed | C3 continue, averaged | R risk first-order (theta=1) | R risk exact-change (theta=5) | L exact line search |
|---|---|---|---|---|---|---|
| 1 | 38.9% | 38.7% | 38.7% | 38.4% | 38.4% | 38.5% |
| 2 | 38.7% | 38.4% | 38.5% | 38.4% | 38.3% | 38.4% |
| 5 | 39.0% | 38.5% | 38.6% | 38.4% | 38.2% | 38.5% |
| 20 | 38.8% | 38.5% | 38.5% | 39.0% | 38.9% | 39.0% |
