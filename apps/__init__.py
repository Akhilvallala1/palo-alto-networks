"""Application layer.

Nothing here may import a vendor SDK. Application code reaches a model only
through the Conduit gateway's HTTP API, which is the epic's load-bearing
constraint (AC-16) and is enforced by `tests/test_no_vendor_imports.py`.
"""
