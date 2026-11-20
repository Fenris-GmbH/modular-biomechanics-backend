"""Helpers for normalizing keys in serialized output dictionaries."""


def lowercase_output_keys(value):
    """Return a recursive copy with all string dictionary keys lower-cased.

    Raise ``ValueError`` rather than silently discard data if lower-casing
    makes two keys identical.
    """
    if isinstance(value, dict):
        lowered = {}
        for key, item in value.items():
            lowered_key = key.lower() if isinstance(key, str) else key
            if lowered_key in lowered:
                raise ValueError(
                    f"Output keys collide after lower-casing: {lowered_key!r}"
                )
            lowered[lowered_key] = lowercase_output_keys(item)
        return lowered
    if isinstance(value, list):
        return [lowercase_output_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(lowercase_output_keys(item) for item in value)
    return value
