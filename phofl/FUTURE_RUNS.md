# Future runs

## Queue status

The one-item queue is **empty**. FedAPEN/linear completed; no run is currently
queued.

## Completed: FedAPEN / linear

| Field | Value |
|---|---|
| Method | FedAPEN |
| Pattern | linear |
| Seed | 0 (one seed; matches the existing four runs) |
| Scale | Half-scale: CIFAR-10, 10 clients, 2,500 train / 500 test per client |
| Training | 60 rounds, four CNNs (`HtM4`), 1 local epoch |
| Cost | 41.2 min on the local RTX 3060 6 GB |
| Result | client-FM 0.0839; class gap 52.9 points; support-filtered gap 41.6 points; FM rank 3, class-gap rank 5 (inverted) |
| Scope action | >30-point gap → field-wide across the two MH-FL methods tested |

No OOM at batch 64; the batch-32 retry was not needed.

## Not queued — full-scale audit

| Field | Value |
|---|---|
| Methods | PHP-FL, Standalone, FedAPEN |
| Data | CIFAR-10, 20 clients, full data |
| Training | 100 rounds, 3 local epochs, 3 seeds, four CNNs |
| Estimated cost | ~40-60 T4 GPU-hours |
| Why | Test whether the 44-53-point class spread and the rank disagreement survive at the paper's training budget. |

This is a class-fairness follow-up, not a seed extension of the closed C
participation hypothesis. Do not execute without separate approval.
