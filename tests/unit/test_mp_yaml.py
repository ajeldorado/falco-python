import falco
import numpy as np
import pytest

parse = falco.config.ModelParameters.from_yaml


def test_basic_parse():
    result = parse("""
        a: 5
        b: Hi
        c:
            field: false
    """)

    assert result.a == 5
    assert result.b == "Hi"
    assert not result.c.field

    assert isinstance(result, falco.config.Object)
    assert isinstance(result.c, falco.config.Object)


def test_probe():
    result = parse("""
        probe: !Probe
            Npairs: 20
            extra_field: Extra
    """)

    assert result.probe.radius == 12
    assert result.probe.Npairs == 20
    assert result.probe.extra_field == "Extra"

    assert isinstance(result.probe, falco.config.Probe)


def test_basic_eval():
    result = parse("""
        expr: !eval 2+2
        not_expr: 2+2
    """)

    assert isinstance(result.data['expr'], falco.config.Eval)
    assert result.data['expr'] != 4
    assert result.expr == 4

    assert isinstance(result.data['not_expr'], str)
    assert result.not_expr == "2+2"


def test_numpy_eval():
    result = parse("""
    mat: !eval np.ones([2, 2])
    """)

    assert np.array_equal(result.mat, np.ones([2, 2]))


def test_falco_eval():
    result = parse("""
        val: !eval falco.INFLUENCE_BMC_2K
    """)

    assert result.val == falco.INFLUENCE_BMC_2K


def test_math_module_eval():
    result = parse("""
        val: !eval math.sin(math.pi)
    """)

    import math
    assert result.val == math.sin(math.pi)


def test_dependency():
    result = parse("""
        a: !eval 2+2
        b: !eval mp.a * 3
        c: !eval mp.a + mp.b
    """)

    # testing in reverse order on purpose
    assert result.c == 16
    assert result.b == 12
    assert result.a == 4


def test_cache():
    result = parse("""
        val: !eval 2+2
        
        dependent_val: !eval mp.val + 2
    """)

    global invocation_count
    invocation_count = 0

    # look away, kids

    original_eval = result.data['val'].evaluate

    def new_eval():
        global invocation_count
        invocation_count += 1
        return original_eval()

    result.data['val'].evaluate = new_eval

    assert result.dependent_val == 6
    assert invocation_count == 1

    assert result.val == 4
    assert invocation_count == 1


def test_circular_dependency():
    result = parse("""
        a: !eval mp.b
        b: !eval mp.a
    """)

    with pytest.raises(Exception):
        assert result.a == 1
