"""Military exercises, drills and deployments in the Indo-Pacific, collected continuously.

Reads the Observatory site's news feed (about 48 sources) and official press
releases, has an open model (llama.cpp on Actions, no key) read each candidate,
checks every answer against the source text, and merges the answers into events
(region, forces, content, period, sources). See exercises/run.py.
"""
