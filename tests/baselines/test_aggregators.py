from baselines.aggregators import *
import pytest
from collections import Counter


def test_percentiles():
    # Test with integer values
    values = [1, 2, 3, 4, 5]
    result = percentiles(values)
    assert result == {'min': 1, 'max': 5, 'median': 3, 'p25': 2, 'p75': 4, 'p90': 5, 'p95': 5, 'p99': 5, 'mean': 3}

    # Test with floating point values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = percentiles(values)
    assert result == {'min': 1.0, 'max': 5.0, 'median': 3.0, 'p25': 2.0, 'p75': 4.0, 'p90': 5.0, 'p95': 5.0, 'p99': 5.0, 'mean': 3.0}

    # Test with negative values
    values = [-1, -2, -3, -4, -5]
    result = percentiles(values)
    assert result == {'min': -5, 'max': -1, 'median': -3, 'p25': -4, 'p75': -2, 'p90': -1, 'p95': -1, 'p99': -1, 'mean': -3}

    # Test with empty list
    values = []
    with pytest.raises(IndexError):
        percentiles(values)

    # Test with one value
    values = [5]
    result = percentiles(values)
    assert result == {'min': 5, 'max': 5, 'median': 5, 'p25': 5, 'p75': 5, 'p90': 5, 'p95': 5, 'p99': 5, 'mean': 5}

    # Test with None as values
    values = None
    with pytest.raises(TypeError):
        percentiles(values)

    # Test with values not as list
    values = 5
    with pytest.raises(TypeError):
        percentiles(values)


def test_histogram():
    # Test with integer values
    values = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    result = histogram(values)
    assert isinstance(result, dict)
    assert len(result) == 10

    # Test with string values
    values = ['a', 'b', 'b', 'c', 'c', 'c']
    result = histogram(values)
    assert result == Counter(values)

    # Test with negative values
    values = [-1, -2, -2, -3, -3, -3]
    result = histogram(values)
    assert isinstance(result, dict)
    assert len(result) == 10

    # Test with floating point values
    values = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    result = histogram(values)
    assert isinstance(result, dict)
    assert len(result) == 10

    # Test with empty list
    values = []
    with pytest.raises(IndexError):
        histogram(values)

    # Test with one value
    values = [5]
    result = histogram(values)
    assert isinstance(result, dict)
    assert len(result) == 10

    # Test with None as values
    values = None
    with pytest.raises(TypeError):
        histogram(values)

    # Test with values not as list
    values = 5
    with pytest.raises(TypeError):
        histogram(values)
