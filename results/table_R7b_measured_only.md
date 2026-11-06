| Strategy | Model | Patients with all inputs measured, n | R2, all patients | R2, measured inputs only (95% CI) | RMSE, all | RMSE, measured only |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | Linear regression | 98 | 0.865 | 0.865 (0.789–0.907) | 3.08 | 3.08 |
| C0 | MLP (capacity-matched) | 98 | 0.817 | 0.817 (0.678–0.892) | 3.60 | 3.60 |
| C0 | LSTM (per-time-point) | 98 | 0.808 | 0.808 (0.705–0.865) | 3.67 | 3.67 |
| C0+C0.5 | Linear regression | 92 | 0.878 | 0.879 (0.810–0.918) | 2.93 | 2.96 |
| C0+C0.5 | MLP (capacity-matched) | 92 | 0.834 | 0.836 (0.733–0.894) | 3.41 | 3.43 |
| C0+C0.5 | LSTM (per-time-point) | 92 | 0.737 | 0.744 (0.599–0.818) | 4.30 | 4.30 |
| C0+C6 | Linear regression | 93 | 0.964 | 0.963 (0.931–0.980) | 1.59 | 1.59 |
| C0+C6 | MLP (capacity-matched) | 93 | 0.939 | 0.937 (0.883–0.962) | 2.07 | 2.08 |
| C0+C6 | LSTM (per-time-point) | 93 | 0.918 | 0.913 (0.877–0.943) | 2.41 | 2.45 |
| C0.5+C6 | Linear regression | 87 | 0.942 | 0.945 (0.914–0.959) | 2.02 | 1.97 |
| C0.5+C6 | MLP (capacity-matched) | 87 | 0.895 | 0.899 (0.831–0.927) | 2.73 | 2.68 |
| C0.5+C6 | LSTM (per-time-point) | 87 | 0.884 | 0.887 (0.828–0.918) | 2.86 | 2.83 |
| C0+C0.5+C6 | Linear regression | 87 | 0.973 | 0.974 (0.955–0.985) | 1.39 | 1.35 |
| C0+C0.5+C6 | MLP (capacity-matched) | 87 | 0.949 | 0.950 (0.908–0.970) | 1.90 | 1.88 |
| C0+C0.5+C6 | LSTM (per-time-point) | 87 | 0.900 | 0.901 (0.854–0.927) | 2.65 | 2.64 |
