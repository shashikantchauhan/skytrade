"""Dashboard (``webapp.py``) internals, split by responsibility -- Phase 16
of `projectedPlann.md` (see docs/architecture/000-audit.md).

Scoped honestly: only the auth/session service (``web.services.auth``) is
split out so far -- it's the one piece every other route already depends
on regardless of how routes themselves are organized, and the most
security-sensitive code in the file, so it benefits most from living on
its own with a focused module docstring. The remaining ~30 routes stay in
``webapp.py`` for now: unlike ``application/signal_pipeline.py``'s split
(Phase 8), there is no existing route-level test suite for this file to
catch a mistake in a larger mechanical move, and a live, already-in-use
dashboard is the wrong place to attempt one without that safety net. See
``docs/architecture/000-audit.md``'s stage notes for the full reasoning.
"""
