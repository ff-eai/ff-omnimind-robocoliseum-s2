# FF Omnimind: RoboColiseum Season II, job 1948

Code of RoboColiseum Season II evaluation job 1948 (UUID `ad445254-1b86-4fe2-a4cc-f4aaa26829f3`, board `v2`), entry FF Omnimind of
FF EAI Robotics Ecosystem Inc., platform score 0.2746. Provided only for the organizers' review; see
`LICENSE-FF.txt`. The system declaration (rule 5.5.1) is provided as a separate document.

- `examples/g2s2/programs/`: one program per task, selected by task name. The cross-embodiment tasks link to six
  shared programs; `default.py` runs for a task without a program (none in this job).
- `src/cap_harness/g2s2/`: the worker, a GenieSim v3.6 policy server.
- `src/cap_harness/providers/sam3/`: the SAM 3 segmentation service used by the programs.

**Weights:** none of our own. The only learned model is SAM 3 (Hugging Face `facebook/sam3`), used unmodified.

## Run

1. **SAM 3 service** (one GPU). Install `requirements-sam3.txt` and the SAM 3 code
   (https://github.com/Max-Fu/sam3 at `6fe87d64a5beb9084923d7a9e002741178635b09`, `pip install -e`). Checkpoint:
   `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`.

   ```bash
   PYTHONPATH=src python -m cap_harness.providers.sam3.service --host 127.0.0.1 --port 8114
   ```

2. **Workers** (CPU). Install `requirements-worker.txt`. Job 1948 ran four:

   ```bash
   for p in 9101 9102 9103 9104; do
     PYTHONPATH=src CAP_HARNESS_SAM3_URL=http://127.0.0.1:8114 CAP_HARNESS_S2_CUROBO_URL=http://127.0.0.1:9 \
       python -m cap_harness.g2s2.serve --host 127.0.0.1 --port $p \
         --programs examples/g2s2/programs --out runs/w$p --sam3-url http://127.0.0.1:8114 \
         --keepalive-s 90 --keepalive-runtime newton=90 &
   done
   ```

3. **Simulator.** Point a GenieSim v3.6 benchmark client at a worker (`--infer-host=127.0.0.1:<port>`), or connect
   the tunnels of a platform job to the four ports (one tunnel per worker).

Licenses: `LICENSE-FF.txt` for the files with the FF copyright line; `LICENSE` and `NOTICE` for the other files.
