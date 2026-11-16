"""DynaMeta's validation scripts (run with ``python -m validation.<name>`` or ``python -m validation.run_all``).

A regular package, so that another project's ``validation`` package on the same ``sys.path`` cannot shadow
this one: a namespace package (no ``__init__.py``) loses to any regular package of the same name anywhere on
the path, whatever the order. Not part of the installed distribution (``packages.find`` includes only
``dynameta*``).
"""
