"""Offline export of the static assets the website serves.

Nothing in here runs at request time. Each module is a one-shot build step that
reads the research artefacts (BIOM tables, checkpoints, embedding text files)
and writes binaries into ``data/web/``. See ``script/web_export/README.md``.
"""
