# ylstm_architecture

Experiment group for the hybrid architectures and the published baselines.
The published LSTM, GRU, Transformer, Mamba, sLSTM, mLSTM, and xLSTM models
are the root registry. Which directories back which thesis experiment is
recorded in `experiments/thesis_manifest.yaml`.

```bash
python -m ml train -c <path-to-setup-yaml>
```

## Names

| Code name | What it is |
|---|---|
| `hlstm` | parallel sLSTM and mLSTM branches in one block |
| `ylstm` | yLSTM cell, bare stack |
| `ylstm_hb` | yLSTM cell in the single-branch hLSTM block |
| `yhlstm` | yLSTM on the matrix-memory branch of the hLSTM block |
| `xlstm_hb` | standard cells, alternating, in the single-branch hLSTM block |
| `ylstm_sb` / `ylstm_mb` | yLSTM cell in the sLSTM or mLSTM block |
| `slstm_hb` / `mlstm_hb` | standard cell in the single-branch hLSTM block |
| `slstm_bare` / `mlstm_bare` | standard cell, bare stack |
| `ylstm_hb_c0` … `ylstm_hb_c8` | yLSTM at chunk sizes T, 1, 2, 4, and 8 |
