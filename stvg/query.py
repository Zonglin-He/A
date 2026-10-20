"""Query-only subject parsing used by the frozen Source."""
class QuerySubjectParser:
    """Query-only Stanza dependency parser, explicitly not official CoreNLP."""
    def __init__(self, model_dir):
        import stanza
        self.version = stanza.__version__
        self.nlp = stanza.Pipeline("en", dir=str(model_dir), package=None,
            processors={"tokenize": "ewt", "mwt": "ewt", "pos": "ewt_nocharlm",
                        "lemma": "ewt_nocharlm", "depparse": "ewt_nocharlm"},
            use_gpu=False, download_method=None, verbose=False)

    def __call__(self, query):
        words = [w for s in self.nlp(query).sentences for w in s.words]
        chosen, rule = None, None
        # A wh-determiner identifies the queried noun even when it is an object.
        for w in words:
            if w.lemma.lower() in {"which", "what"} and w.deprel == "det":
                chosen = next((v for v in words if v.id == w.head), None)
                if chosen:
                    rule = "wh_determiner_head"
                    break
        if chosen is None:
            for w in words:
                if w.lemma.lower() in {"who", "whom"}:
                    return {"subject": "person", "rule": "explicit_who", "parser_version": self.version}
            if any(w.lemma.lower() == "what" for w in words):
                return {"subject": "", "rule": "unknown_what_no_category_in_query", "parser_version": self.version}
            chosen = next((w for w in words if w.deprel.startswith("nsubj") and w.upos in {"NOUN", "PROPN"}), None)
            rule = "dependency_nominal_subject"
        if chosen is None:
            chosen = next((w for w in words if w.upos in {"NOUN", "PROPN"}), None)
            rule = "first_noun_fallback"
        return {"subject": chosen.lemma.lower() if chosen else "", "rule": rule if chosen else "empty_prefix_fallback",
                "parser_version": self.version}
