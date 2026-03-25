#!/usr/bin/env python
"""Test script for FALCO serialization."""
import numpy as np
import falco
import os
import tempfile

def test_basic_serialization():
    """Test basic save/load functionality."""
    print("Testing basic serialization...")

    # Create a test Object with various data types
    obj = falco.config.Object()
    obj.data['scalar'] = 42
    obj.data['string'] = 'test'
    obj.data['array'] = np.array([1, 2, 3, 4, 5])
    obj.data['matrix'] = np.random.rand(10, 10)
    obj.data['complex_array'] = np.array([1+2j, 3+4j, 5+6j])
    obj.data['nested'] = {'a': 1, 'b': [2, 3, 4]}

    # Create nested Object
    nested_obj = falco.config.Object()
    nested_obj.data['value'] = 100
    nested_obj.data['array'] = np.array([10, 20, 30])
    obj.data['nested_object'] = nested_obj

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_data')

        # Save
        print(f"  Saving to {filepath}...")
        falco.serialize.save(obj, filepath)

        # Verify files were created
        assert os.path.exists(filepath + '.h5'), "HDF5 file not created"
        assert os.path.exists(filepath + '.json'), "JSON file not created"
        print("  ✓ Files created")

        # Load
        print(f"  Loading from {filepath}...")
        loaded = falco.serialize.load(filepath)

        # Verify data
        assert loaded.scalar == 42, "Scalar mismatch"
        assert loaded.string == 'test', "String mismatch"
        assert np.allclose(loaded.array, obj.data['array']), "Array mismatch"
        assert np.allclose(loaded.matrix, obj.data['matrix']), "Matrix mismatch"
        assert np.allclose(loaded.complex_array, obj.data['complex_array']), "Complex array mismatch"
        assert loaded.nested['a'] == 1, "Nested dict mismatch"
        assert loaded.nested_object.value == 100, "Nested object mismatch"
        assert np.allclose(loaded.nested_object.array, np.array([10, 20, 30])), "Nested object array mismatch"

        print("  ✓ All data verified")

    print("✓ Basic serialization test passed!\n")


def test_load_as_dict():
    """Test loading as plain dicts."""
    print("Testing load as dict...")

    obj = falco.config.Object()
    obj.data['value'] = 123
    obj.data['array'] = np.array([1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_dict')

        # Save
        falco.serialize.save(obj, filepath)

        # Load as dict
        loaded = falco.serialize.load(filepath, reconstruct_objects=False)

        # Verify it's a dict, not an Object
        assert isinstance(loaded, dict), "Should be dict"
        assert not isinstance(loaded, falco.config.Object), "Should not be Object"
        assert loaded['value'] == 123, "Value mismatch"
        assert np.allclose(loaded['array'], np.array([1, 2, 3])), "Array mismatch"

        print("  ✓ Dict loading verified")

    print("✓ Load as dict test passed!\n")


def test_large_arrays():
    """Test with larger arrays to verify compression."""
    print("Testing large arrays...")

    obj = falco.config.Object()
    obj.data['large_array'] = np.random.rand(1000, 1000)
    obj.data['large_complex'] = np.random.rand(500, 500) + 1j * np.random.rand(500, 500)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_large')

        # Save
        falco.serialize.save(obj, filepath)

        # Load
        loaded = falco.serialize.load(filepath)

        # Verify
        assert np.allclose(loaded.large_array, obj.data['large_array']), "Large array mismatch"
        assert np.allclose(loaded.large_complex, obj.data['large_complex']), "Large complex array mismatch"

        # Check file size (should be compressed)
        h5_size = os.path.getsize(filepath + '.h5')
        uncompressed_size = obj.data['large_array'].nbytes + obj.data['large_complex'].nbytes
        print(f"  Uncompressed: {uncompressed_size / 1e6:.2f} MB")
        print(f"  Compressed HDF5: {h5_size / 1e6:.2f} MB")
        print(f"  Compression ratio: {uncompressed_size / h5_size:.2f}x")

    print("✓ Large array test passed!\n")


if __name__ == '__main__':
    print("=" * 60)
    print("FALCO Serialization Test Suite")
    print("=" * 60 + "\n")

    try:
        test_basic_serialization()
        test_load_as_dict()
        test_large_arrays()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
