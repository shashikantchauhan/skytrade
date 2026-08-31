"""Experimental AlphaEngine variants -- never wired into the live pipeline
or ``application/backtest.py``'s production path. See each module's own
docstring for what it tests and why. ``alpha_engine.py`` itself is never
modified (hard constraint, see NOTES.md) -- everything here is a separate
subclass reusing its private vectorized helpers.
"""
