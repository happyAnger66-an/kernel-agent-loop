def launch_config(shape):
    rows = shape.dims["rows"]
    hidden = shape.dims["hidden"]
    return {
        "kernel": "rmsnorm",
        "grid": (rows, 1, 1),
        "block": (256, 1, 1),
        "smem": 0,
        "args": ["x", "weight", "y", "rows", "hidden"],
    }
