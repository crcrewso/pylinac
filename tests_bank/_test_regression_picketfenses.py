# Regression testing to catch changes in analysis values as the imaging algorithms change
from pylinac import picketfence
import pytest
import os


# key and value for results dictionary for expected exact answers with no margin
exactTests = [
    ('tolerance_mm', 0.15),
    ('number_of_pickets', 10),
    ('passed', True),
    ('max_error_picket', 1),
    ('max_error_leaf', 42),
    ('failed_leaves', []),

]


# key and value for results dictionary for expected float answers with clinical test tolerances
clinicalTests = [
    ('absolute_median_error_mm', 0.026, 2e-3),
    ('max_error_mm', 0.10, 1e-1),
    ('mean_picket_spacing_mm', 15, 1e-2),
    ('mlc_skew', -0.06, 1e-2)

]

# key and value for results dictionary for expected float answers with tighter test tolerances for regression testing
regressionTests = [
    ('absolute_median_error_mm', 0.026807700244109167, 1e-6),
    ('max_error_mm', 0.10013319651342273, 1e-6),
    ('mean_picket_spacing_mm', 15.009392906700404, 1e-6),
    ('mlc_skew', -0.06350760797506376, 1e-6)
]


@pytest.fixture(scope='session')
def good_PF():
    rootdir = os.getcwd()
    dir = os.path.join(rootdir, "Pylinac", "Resources", "MLC")
    file = os.path.join(dir, "Solstice-m01_d03_2024-Dynamic_MLC_000.dcm")
    pf = picketfence.PicketFence(file)
    pf.analyze(tolerance=0.15)
    return pf.results_data(as_dict=True)


@pytest.mark.parametrize("test_key, test_value", exactTests)
def test_int(good_PF, test_key, test_value):
    assert good_PF[test_key] == test_value


@pytest.mark.parametrize("test_key, test_value, test_tol", clinicalTests)
def test_clinical(good_PF, test_key, test_value, test_tol):
    assert good_PF[test_key] == pytest.approx(
        expected=test_value, abs=test_tol)


@pytest.mark.parametrize("test_key, test_value, test_tol", regressionTests)
def test_regression(good_PF, test_key, test_value, test_tol):
    assert good_PF[test_key] == pytest.approx(
        expected=test_value, abs=test_tol)
