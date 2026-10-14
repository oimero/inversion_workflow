"""Physical numerical kernels and relation values.

Import a concrete module such as :mod:`cup.physics.numpy_backend`,
:mod:`cup.physics.torch_backend`, or :mod:`cup.physics.relations` explicitly.
PyTorch remains optional for callers that only use the NumPy implementation.
Finite-run response support helpers live in :mod:`cup.physics.support` and
are NumPy-only so they can be shared by differentiable and reporting paths.
"""
