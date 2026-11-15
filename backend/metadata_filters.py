
def matches_metadata(metadata: dict, filters: dict | None) -> bool:
    """Evaluate Chroma-style metadata filters against a document's metadata."""
    if not filters:
        return True

    for key, condition in filters.items():
        if key == "$and":
            if not all(matches_metadata(metadata, item) for item in condition):
                return False
            continue

        if key == "$or":
            if not any(matches_metadata(metadata, item) for item in condition):
                return False
            continue

        actual = metadata.get(key)

        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator == "$eq" and actual != expected:
                    return False
                elif operator == "$ne" and actual == expected:
                    return False
                elif operator == "$gt" and not (actual is not None and actual > expected):
                    return False
                elif operator == "$gte" and not (actual is not None and actual >= expected):
                    return False
                elif operator == "$lt" and not (actual is not None and actual < expected):
                    return False
                elif operator == "$lte" and not (actual is not None and actual <= expected):
                    return False
                elif operator == "$in" and actual not in expected:
                    return False
                elif operator == "$nin" and actual in expected:
                    return False
                elif operator not in {
                    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"
                }:
                    raise ValueError(f"Unsupported metadata operator: {operator}")
        elif actual != condition:
            return False

    return True
