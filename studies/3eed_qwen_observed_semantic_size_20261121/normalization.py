"""Close the semantic interface under the fixed taxonomy before task scoring."""
from study import TAXONOMY, parse_json, parse_semantic, size_requests, text_hash

SUBTYPE_PARENT = {subtype: category for category, types in TAXONOMY.items() for subtype in types
                  if subtype != "unknown" and sum(subtype in v for v in TAXONOMY.values()) == 1}


def normalize(answer):
    parsed = parse_semantic(answer)
    if parsed is not None:
        return parsed, "valid"
    data = parse_json(answer)
    if data is None or set(data) != {"category", "subtype"} or not all(isinstance(x, str) for x in data.values()):
        return None, "invalid"
    category, subtype = data["category"], data["subtype"]
    if category in TAXONOMY:
        return [category, "unknown"], "coarse_only"
    if category in SUBTYPE_PARENT:
        parent = SUBTYPE_PARENT[category]
        if subtype in SUBTYPE_PARENT and subtype != category:
            return [parent, "unknown"], "conflicting_subtype"
        return [parent, category], "subtype_promoted"
    return None, "invalid"


def resolve(frame, spans, raw):
    """Find missing size prompts and the exact archived responses that can be reused."""
    labels, statuses = {}, {}
    for arm, record in raw["semantic"].items():
        labels[arm], statuses[arm] = normalize(record["answer"])
    prompts, routes = size_requests(frame["text"], spans, labels)
    existing = {r["prompt_sha256"]: r for r in raw["sizes"].values()}
    reused = {key: existing[text_hash(prompt)] for key, prompt in prompts.items() if text_hash(prompt) in existing}
    missing = {key: prompt for key, prompt in prompts.items() if key not in reused}
    return labels, statuses, routes, reused, missing
