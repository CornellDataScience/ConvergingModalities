"""prh: tools for testing representational convergence (Platonic Representation Hypothesis)."""
from .schema import SCHEMA_VERSION, SchemaError, add_record, new_run, read, validate, write

__all__ = ['SCHEMA_VERSION', 'SchemaError', 'add_record', 'new_run', 'read', 'validate', 'write']
__version__ = '0.1.0'
