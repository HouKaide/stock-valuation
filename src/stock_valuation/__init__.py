"""Stock valuation package.

The package root exposes shared boundaries such as the CLI entrypoint and
cross-layer error types without owning business logic directly.
"""

__all__ = [
    "cli",
    "errors",
]
