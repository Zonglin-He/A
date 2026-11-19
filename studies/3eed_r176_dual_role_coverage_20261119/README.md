# R176 Dual-role coverage and frozen four-slot retrieval

The diagnosis traces missing A256 hypotheses, shortlist misses, and output misses. Source-held verified boxes and ordinary target reviewed references are scored separately. The retriever intervention uses the frozen R175 pair-only decoder on the same 18,620 rows.

In the registered workspace, run diagnose176.py, summarize176.py --gate, retrieve176.py, score176.py, audit176.py, and summarize176.py in that order. The retrieval process reads no grounding labels. The diagnostics require the existing private benchmark and R175 caches. Only anonymous scalar outcomes, protocols, implementation and audit receipts are published.
