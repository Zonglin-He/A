# Source-Supervised Relational Support Feedback

R175 uses each of three frozen single-platform 3EED models and same-source RGB features. Source-only labelled role pairs train the candidate decoder. Target checkpoints are locked before paired prediction on the same 18,620 ordinary Validation direction records.

The locked implementation and controls are under own_code. The workspace commands are source175.py, features175.py, train175.py, predict175.py, score175.py, audit175.py, mechanism175.py and report175.py. The preparation and training commands take the three source names. Frozen feature caches, raw annotations, checkpoint weights and coordinates are private prerequisites. The public scalar files retain both signs of all observed differences. See REPORT_ZH.md for details.
