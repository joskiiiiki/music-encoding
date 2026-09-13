"""Backend for the MTG-Jamendo similarity explorer.

Reads only the exported artifacts in ``webapp/data/`` (sqlite + .npz) -- never the
model package, chromadb, or torch at request time.
"""
