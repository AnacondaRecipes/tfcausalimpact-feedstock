import pytest
import numpy as np
import pandas as pd
from causalimpact import CausalImpact

@pytest.fixture
def sample_data():
    # Generate sample data for testing
    np.random.seed(42)
    pre_period = [0, 99]
    post_period = [100, 199]
    data = np.random.randn(200, 2)
    data[100:, 0] += 2  # Introduce a change in the first column after the intervention
    df = pd.DataFrame(data, columns=['y', 'x1'])
    return df, pre_period, post_period

def test_causalimpact_initialization(sample_data):
    df, pre_period, post_period = sample_data
    # Initialize the CausalImpact model
    ci = CausalImpact(df, pre_period, post_period)
    assert isinstance(ci, CausalImpact), "CausalImpact initialization failed"

def test_summary_method(sample_data):
    df, pre_period, post_period = sample_data
    ci = CausalImpact(df, pre_period, post_period)
    # Test the summary method
    summary = ci.summary()
    assert isinstance(summary, str), "Summary method did not return a string"

def test_p_value_access(sample_data):
    df, pre_period, post_period = sample_data
    ci = CausalImpact(df, pre_period, post_period)
    # Access the p-value directly
    p_value = ci.p_value
    assert isinstance(p_value, float), "P-value is not a float"
    assert 0 <= p_value <= 1, "P-value is out of range"