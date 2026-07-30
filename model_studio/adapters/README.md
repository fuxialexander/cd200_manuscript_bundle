# Model adapter contract

The studio deliberately separates model reconstruction from perturbation
analysis and the browser API.

Every adapter subclasses `SequenceModelAdapter` and:

1. Accepts canonical one-hot DNA shaped `(batch, length, 4)`.
2. Loads one checkpoint family and owns its tensor layout/device handling.
3. Returns named NumPy arrays shaped `(batch, output_length)`.
4. Publishes `ModelMetadata`, including output heads and fixed input length when
   relevant.

Register an adapter class in `ModelAdapterRegistry`. No server, deletion engine,
or frontend changes are required for a model that fits this contract. If the
model needs architecture-specific supporting artifacts, declare them in the
descriptor's `option_schema`; the frontend builds those controls from the API.

The GET v2 pool16 transformer adapter is not published in this bundle yet.
