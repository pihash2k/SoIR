"""
Feature extractors for the retrieval system.
"""

__all__ = ['BaseExtractor', 'DinoV2Extractor', 'DinoV2MIExtractor']

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'BaseExtractor':
        from extractors.base_extractor import BaseExtractor
        return BaseExtractor
    elif name == 'DinoV2Extractor':
        from extractors.dinov2_extractor import DinoV2Extractor
        return DinoV2Extractor
    elif name == 'DinoV2MIExtractor':
        from extractors.dinov2_mi_extractor import DinoV2MIExtractor
        return DinoV2MIExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
