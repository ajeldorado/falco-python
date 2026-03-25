# FALCO Serialization

FALCO now uses a secure HDF5 + JSON serialization format instead of pickle.

## Why the change?

- **Security**: Pickle can execute arbitrary code when loading, making it vulnerable to malicious files
- **Compatibility**: HDF5 + JSON are standard, language-agnostic formats
- **Performance**: HDF5 is optimized for large numerical arrays

## Usage

### Saving data

```python
import falco

# Save output data
falco.serialize.save(out, 'output_data')
# Creates: output_data.h5 and output_data.json

# Save model parameters
falco.serialize.save(mp, 'model_params')
# Creates: model_params.h5 and model_params.json
```

### Loading data

```python
import falco

# Load data
out = falco.serialize.load('output_data')
mp = falco.serialize.load('model_params')

# You can specify either .h5, .json, or just the base name
out = falco.serialize.load('output_data.h5')  # Same result
```

## Backward Compatibility

The `plot_trial_output_from_pickle()` function still supports legacy `.pkl` files:

```python
import falco.plot

# Works with both old and new formats
falco.plot.plot_trial_output_from_pickle('data.pkl')  # Old format
falco.plot.plot_trial_output_from_pickle('data.h5')   # New format
```

## File Format

Each save operation creates two files:

1. **`.h5` file**: Contains all numpy arrays in HDF5 format (compressed)
2. **`.json` file**: Contains metadata, structure, and references to arrays

Both files are needed to reconstruct the original object.

## Advanced Usage

### Loading as plain dicts

If you don't want to reconstruct `Object` instances:

```python
data = falco.serialize.load('output_data', reconstruct_objects=False)
# Returns plain nested dicts instead of Object instances
```

## Migration Guide

### For existing code:

**Old:**
```python
import pickle

# Saving
with open('data.pkl', 'wb') as f:
    pickle.dump(out, f)

# Loading
with open('data.pkl', 'rb') as f:
    out = pickle.load(f)
```

**New:**
```python
import falco

# Saving
falco.serialize.save(out, 'data')

# Loading
out = falco.serialize.load('data')
```

## Notes

- The WFSC loop now automatically saves data in the new format
- Files are named with base name only (no `.pkl` extension)
- The new format creates `.h5` and `.json` pairs
- Complex numbers are supported
- Nested `Object` structures are fully supported
