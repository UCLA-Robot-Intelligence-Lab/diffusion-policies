from typing import Dict, Callable, Any

def dict_apply(obj: Dict, fn: Callable) -> Dict:
    """
    Apply a function to all values in a dictionary. Works recursively.
    
    Args:
        obj: Dictionary to apply function to
        fn: Function to apply to each value
        
    Returns:
        Dictionary with function applied to all values
    """
    if isinstance(obj, dict):
        return {k: dict_apply(v, fn) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(dict_apply(v, fn) for v in obj)
    else:
        return fn(obj) 