"""Simple recursive dict subset checker."""


def is_dict_subset(subset: dict, superset: dict) -> bool:
    """
    Check if subset dict is a subset of superset dict.

    Rules:
    - All keys from subset must exist in superset
    - Values must be equal (recursively for nested dicts)
    - Lists must be exactly equal (order matters)
    - Empty dict is subset of any dict

    Args:
        subset: Dictionary to check if it's a subset
        superset: Dictionary to check against

    Returns:
        bool: True if subset is a subset of superset
    """
    # Type check
    if not isinstance(subset, dict) or not isinstance(superset, dict):
        raise TypeError("Both arguments must be dictionaries")

    # Check all keys from subset exist in superset with same values
    for key, subset_value in subset.items():
        if key not in superset:
            return False

        superset_value = superset[key]

        # Both are dicts - recursive check
        if isinstance(subset_value, dict) and isinstance(superset_value, dict):
            if not is_dict_subset(subset_value, superset_value):
                return False

        # Both are lists - check length and recursively check elements
        elif isinstance(subset_value, list) and isinstance(superset_value, list):
            if len(subset_value) != len(superset_value):
                return False

            for _, (sub_item, super_item) in enumerate(
                zip(subset_value, superset_value, strict=False)
            ):
                # If both items are dicts, check subset relationship
                if isinstance(sub_item, dict) and isinstance(super_item, dict):
                    if not is_dict_subset(sub_item, super_item):
                        return False
                # Otherwise, must be equal
                elif sub_item != super_item:
                    return False

        # Direct equality check for all other types
        elif subset_value != superset_value:
            return False

    return True
