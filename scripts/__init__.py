"""Standalone CLI tools invoked as `python scripts/<name>.py ...`.

Marked as a regular package (not an implicit namespace package) so
setuptools.packages.find can discover and install it, and so tests can
import from it as `scripts.<module>`. See CHANGELOG.md.
"""
