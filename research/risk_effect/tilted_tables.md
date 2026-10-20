German groups used for the worst-group metric (pooled test rows over 20 splits >= 200): {'G1 unemployed now': 383, 'G3 housing for free': 622, 'G4 age < 25': 926, 'G5 not a foreign worker': 224, 'G6 female': 1832, 'G7 checking account < 0 DM': 1675, 'G8 large and long loan': 917, 'G9 no property': 908, 'G10 male divorced/separated': 306} 

## Level of each arm (mean over 20 splits): accuracy | worst-class error | German: worst-group error

| Dataset | Arm | accuracy | false-negative rate | false-positive rate | worst-class error | worst-group error (German) |
|---|---|---|---|---|---|---|
| german | ERM | 75.4% | 13.4% | 50.8% | 50.8% | 39.7% |
| german | tilt theta=0.5 | 75.2% | 13.6% | 50.7% | 50.7% | 38.6% |
| german | reweight (matched to theta=0.5) | 74.2% | 13.1% | 55.2% | 55.2% | 40.7% |
| german | tilt theta=1 | 75.2% | 13.8% | 50.5% | 50.5% | 38.9% |
| german | reweight (matched to theta=1) | 74.0% | 12.8% | 56.8% | 56.8% | 41.8% |
| german | tilt theta=2 | 75.1% | 13.9% | 50.6% | 50.6% | 38.6% |
| german | reweight (matched to theta=2) | 73.8% | 14.3% | 53.8% | 53.9% | 42.0% |
| german | tilt theta=5 | 75.1% | 13.8% | 50.6% | 50.6% | 38.6% |
| german | reweight (matched to theta=5) | 74.0% | 15.0% | 51.7% | 52.0% | 41.5% |
| german | group-DRO (known groups) | 73.6% | 17.5% | 47.2% | 47.2% | 38.6% |
| compas | ERM | 67.3% | 47.6% | 20.2% | 47.6% | - |
| compas | tilt theta=0.5 | 67.3% | 47.7% | 20.2% | 47.7% | - |
| compas | reweight (matched to theta=0.5) | 67.3% | 43.7% | 23.5% | 43.7% | - |
| compas | tilt theta=1 | 67.3% | 47.8% | 20.1% | 47.8% | - |
| compas | reweight (matched to theta=1) | 67.4% | 43.6% | 23.5% | 43.6% | - |
| compas | tilt theta=2 | 67.3% | 47.8% | 20.1% | 47.8% | - |
| compas | reweight (matched to theta=2) | 67.3% | 44.3% | 23.0% | 44.3% | - |
| compas | tilt theta=5 | 67.3% | 47.8% | 20.1% | 47.8% | - |
| compas | reweight (matched to theta=5) | 67.4% | 44.5% | 22.7% | 44.5% | - |
| pima | ERM | 77.5% | 41.8% | 12.2% | 41.8% | - |
| pima | tilt theta=0.5 | 77.5% | 42.0% | 12.2% | 42.0% | - |
| pima | reweight (matched to theta=0.5) | 76.9% | 46.6% | 10.6% | 46.6% | - |
| pima | tilt theta=1 | 77.4% | 42.3% | 12.2% | 42.3% | - |
| pima | reweight (matched to theta=1) | 76.8% | 45.2% | 11.4% | 45.2% | - |
| pima | tilt theta=2 | 77.2% | 42.6% | 12.2% | 42.6% | - |
| pima | reweight (matched to theta=2) | 76.6% | 48.9% | 9.9% | 48.9% | - |
| pima | tilt theta=5 | 77.2% | 42.8% | 12.2% | 42.8% | - |
| pima | reweight (matched to theta=5) | 76.7% | 47.8% | 10.2% | 47.8% | - |
| heart | ERM | 82.3% | 14.6% | 21.5% | 22.0% | - |
| heart | tilt theta=0.5 | 81.4% | 16.4% | 21.2% | 22.3% | - |
| heart | reweight (matched to theta=0.5) | 80.6% | 19.1% | 19.8% | 26.2% | - |
| heart | tilt theta=1 | 80.9% | 17.5% | 21.1% | 22.5% | - |
| heart | reweight (matched to theta=1) | 79.5% | 21.8% | 18.9% | 28.7% | - |
| heart | tilt theta=2 | 80.6% | 18.0% | 21.1% | 22.7% | - |
| heart | reweight (matched to theta=2) | 80.1% | 21.6% | 17.9% | 26.3% | - |
| heart | tilt theta=5 | 80.3% | 18.4% | 21.3% | 22.9% | - |
| heart | reweight (matched to theta=5) | 80.2% | 20.4% | 19.0% | 26.6% | - |

## Primary: tilted minus matched reweighting (percentage points; negative = tilted better), 95% CI over 20 splits

| Dataset | theta | accuracy gap | worst-class error | worst-group error (German) |
|---|---|---|---|---|
| german | 0.5 | +1.0 [+0.4, +1.6] | -4.6 [-11.4, +1.9] | -2.1 [-4.3, +1.2] |
| german | 1 | +1.2 [+0.6, +2.0] | -6.3 [-13.6, +0.9] | -2.9 [-5.1, +0.5] |
| german | 2 | +1.3 [+0.6, +1.9] | -3.3 [-11.6, +4.2] | -3.4 [-5.7, +0.3] |
| german | 5 | +1.1 [+0.4, +1.9] | -1.4 [-9.0, +6.2] | -2.9 [-4.8, +0.5] |
| compas | 0.5 | -0.0 [-0.2, +0.2] | +4.0 [+2.0, +6.1] | - |
| compas | 1 | -0.0 [-0.3, +0.2] | +4.1 [+2.1, +6.3] | - |
| compas | 2 | -0.0 [-0.2, +0.2] | +3.5 [+1.7, +5.6] | - |
| compas | 5 | -0.0 [-0.3, +0.1] | +3.2 [+1.5, +4.9] | - |
| pima | 0.5 | +0.6 [+0.0, +1.1] | -4.6 [-7.1, -2.3] | - |
| pima | 1 | +0.6 [-0.1, +1.4] | -3.0 [-6.4, +0.4] | - |
| pima | 2 | +0.7 [+0.0, +1.4] | -6.2 [-9.8, -2.5] | - |
| pima | 5 | +0.5 [-0.3, +1.3] | -5.0 [-9.1, -0.6] | - |
| heart | 0.5 | +0.8 [-0.6, +2.3] | -3.9 [-6.8, -1.1] | - |
| heart | 1 | +1.4 [-0.0, +2.8] | -6.2 [-9.0, -3.4] | - |
| heart | 2 | +0.5 [-1.2, +2.3] | -3.6 [-6.9, -0.6] | - |
| heart | 5 | +0.1 [-1.6, +1.9] | -3.7 [-6.7, -0.5] | - |

## Tilted minus plain ERM (percentage points)

| Dataset | theta | accuracy | worst-class error | worst-group error (German) |
|---|---|---|---|---|
| german | 0.5 | -0.1 [-0.4, +0.1] | -0.2 [-0.8, +0.4] | -1.0 [-2.5, +0.5] |
| german | 1 | -0.2 [-0.5, +0.1] | -0.3 [-1.0, +0.3] | -0.8 [-2.1, +0.5] |
| german | 2 | -0.3 [-0.7, +0.0] | -0.2 [-1.1, +0.6] | -1.0 [-2.5, +0.9] |
| german | 5 | -0.3 [-0.7, +0.1] | -0.2 [-1.1, +0.7] | -1.0 [-2.5, +0.9] |
| compas | 0.5 | -0.0 [-0.1, +0.0] | +0.0 [-0.1, +0.2] | - |
| compas | 1 | -0.0 [-0.1, +0.0] | +0.1 [-0.1, +0.3] | - |
| compas | 2 | -0.0 [-0.1, +0.0] | +0.2 [-0.0, +0.3] | - |
| compas | 5 | -0.0 [-0.1, +0.0] | +0.1 [-0.1, +0.3] | - |
| pima | 0.5 | -0.1 [-0.3, +0.2] | +0.2 [-0.5, +0.8] | - |
| pima | 1 | -0.2 [-0.7, +0.3] | +0.4 [-0.6, +1.4] | - |
| pima | 2 | -0.3 [-0.9, +0.3] | +0.8 [-0.4, +2.0] | - |
| pima | 5 | -0.3 [-1.0, +0.2] | +1.0 [-0.3, +2.1] | - |
| heart | 0.5 | -0.9 [-1.5, -0.2] | +0.3 [-0.7, +1.5] | - |
| heart | 1 | -1.4 [-2.0, -0.7] | +0.5 [-0.7, +1.7] | - |
| heart | 2 | -1.7 [-2.4, -1.0] | +0.7 [-0.7, +2.0] | - |
| heart | 5 | -2.0 [-2.8, -1.3] | +0.9 [-0.5, +2.3] | - |

## German: group-DRO (known groups) minus ERM and minus tilted theta=5

- vs ERM: accuracy -1.8 [-2.7, -0.9]; worst-group error -1.1 [-3.9, +1.9]
- vs tilt theta=5: accuracy -1.6 [-2.5, -0.6]; worst-group error -0.1 [-2.5, +2.2]
