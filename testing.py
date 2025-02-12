import numpy as np

def test_gd(func):
    # Test case 1: Basic functionality with simple inputs
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    lr = 0.01

    expected_phi = np.array([0.999, 0.998, 0.997])
    
    phi = func(gradient, prev_phi, lr)
    
    assert np.allclose(phi, expected_phi), f"\033[91mTest case 1 failed. \nExpected theta: {expected_phi}, but got {phi}\033[0m"
    
    # Test case 2: Zero gradient should not change weights
    gradient = np.array([0.0, 0.0, 0.0])
    prev_phi = np.array([1.0, 1.0, 1.0])
    lr = 0.01

    expected_phi = np.array([1.0, 1.0, 1.0])
    
    phi = func(gradient, prev_phi, lr)
    
    assert np.allclose(phi, expected_phi), f"\033[91mTest case 2 failed. \nExpected theta: {expected_phi}, but got {phi}\033[0m"
    
    # Test case 3: High learning rate
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    lr = 1.0

    expected_phi = np.array([0.9, 0.8, 0.7])
    
    phi = func(gradient, prev_phi, lr)
    
    assert np.allclose(phi, expected_phi), f"\033[91mTest case 3 failed. \nExpected theta: {expected_phi}, but got {phi}\033[0m"
    
    # Test case 4: Negative gradient should increase weights
    gradient = np.array([-0.1, -0.2, -0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    lr = 0.01

    expected_phi = np.array([1.001, 1.002, 1.003])
    
    phi = func(gradient, prev_phi, lr)
    
    assert np.allclose(phi, expected_phi), f"\033[91mTest case 4 failed. \nExpected theta: {expected_phi}, but got {phi}\033[0m"
    
    # Test case 5: Large negative gradient with high learning rate
    gradient = np.array([-1.0, -2.0, -3.0])
    prev_phi = np.array([1.0, 1.0, 1.0])
    lr = 0.5
    
    expected_phi = np.array([1.5, 2.0, 2.5])
    
    phi = func(gradient, prev_phi, lr)
    
    assert np.allclose(phi, expected_phi), f"\033[91mTest case 5 failed. \nExpected theta: {expected_phi}, but got {phi}\033[0m"

    print("\033[92mAll tests passed!\033[0m")

def test_nm(func):
    # Test case 1: Basic functionality with simple inputs
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.0, 0.0, 0.0])
    beta = 0.9
    lr = 0.01

    expected_phi = np.array([0.9999, 0.9998, 0.9997])
    expected_m = np.array([0.01, 0.02, 0.03])

    phi, m = func(gradient, prev_phi, prev_m, beta, lr)

    assert np.allclose(phi, expected_phi), f"\033[91mTest case 1 failed. \nExpected theta: {expected_phi}, but got theta: {phi}\033[0m"
    assert np.allclose(m, expected_m),  f"\033[91mTest case 1 failed. \nExpected v: {expected_m}, but got v: {m}\033[0m"
    
    # Test case 2: Zero gradient should not change weights or momentum significantly
    gradient = np.array([0.0, 0.0, 0.0])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.5, 0.5, 0.5])
    beta = 0.9
    lr = 0.01

    expected_phi = np.array([0.9955, 0.9955, 0.9955])
    expected_m = np.array([0.45, 0.45, 0.45])

    phi, m = func(gradient, prev_phi, prev_m, beta, lr)

    assert np.allclose(phi, expected_phi), f"\033[91mTest case 2 failed. \nExpected theta: {expected_phi}, but got theta: {phi}\033[0m"
    assert np.allclose(m, expected_m),  f"\033[91mTest case 2 failed. \nExpected v: {expected_m}, but got v: {m}\033[0m"
    
    
    # Test case 3: High learning rate
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.0, 0.0, 0.0])
    beta = 0.9
    lr = 1.0

    expected_phi = np.array([0.99, 0.98, 0.97])
    expected_m = np.array([0.01, 0.02, 0.03])

    phi, m = func(gradient, prev_phi, prev_m, beta, lr)

    assert np.allclose(phi, expected_phi), f"\033[91mTest case 3 failed. \nExpected theta: {expected_phi}, but got theta: {phi}\033[0m"
    assert np.allclose(m, expected_m),  f"\033[91mTest case 3 failed. \nExpected v: {expected_m}, but got v: {m}\033[0m"
    
    # Test case 4: High momentum coefficient
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.5, 0.5, 0.5])
    beta = 0.99
    lr = 0.01

    expected_phi = np.array([0.99504, 0.99503, 0.99502])
    expected_m = np.array([0.496, 0.497, 0.498])

    phi, m = func(gradient, prev_phi, prev_m, beta, lr)

    assert np.allclose(phi, expected_phi), f"\033[91mTest case 4 failed. \nExpected theta: {expected_phi}, but got theta: {phi}\033[0m"
    assert np.allclose(m, expected_m),  f"\033[91mTest case 4 failed. \nExpected v: {expected_m}, but got v: {m}\033[0m"
    
    # Test case 5: Check with negative gradients
    gradient = np.array([-0.1, -0.2, -0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.0, 0.0, 0.0])
    beta = 0.9
    lr = 0.01

    expected_phi = np.array([1.0001, 1.0002, 1.0003])
    expected_m = np.array([-0.01, -0.02, -0.03])

    phi, m = func(gradient, prev_phi, prev_m, beta, lr)

    assert np.allclose(phi, expected_phi), f"\033[91mTest case 5 failed. \nExpected theta: {expected_phi}, but got theta: {phi}\033[0m"
    assert np.allclose(m, expected_m),  f"\033[91mTest case 5 failed. \nExpected v: {expected_m}, but got v: {m}\033[0m"

    print("\033[92mAll tests passed!\033[0m")

def test_adam(func):
    # Test case 1: Basic functionality with simple inputs
    iteration_num = 1
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.0, 0.0, 0.0])
    prev_v = np.array([0.0, 0.0, 0.0])
    beta = 0.9
    gamma = 0.999
    epsilon = 1e-8
    lr = 0.001
    
    m_expected = np.array([0.01, 0.02, 0.03])
    v_expected = np.array([1e-05, 4e-05, 9e-05])
    phi_expected = np.array([0.99925586, 0.99925586, 0.99925586])
    
    phi, m, v = func(iteration_num, gradient, prev_phi, prev_m, prev_v, beta, gamma, epsilon, lr)
    
    assert np.allclose(m, m_expected), f"\033[91mTest case 1 failed. \nExpected m: {m_expected}, but got m: {m}\033[0m"
    assert np.allclose(v, v_expected), f"\033[91mTest case 1 failed. \nExpected v: {v_expected}, but got m: {v}\033[0m"
    assert np.allclose(phi, phi_expected), f"\033[91mTest case 1 failed. \nExpected theta: {phi_expected}, but got theta: {phi}\033[0m"

    # Test case 2: Check with larger gradients
    iteration_num = 10
    gradient = np.array([1.0, 1.0, 1.0])
    prev_phi = np.array([0.5, 0.5, 0.5])
    prev_m = np.array([0.1, 0.1, 0.1])
    prev_v = np.array([0.01, 0.01, 0.01])
    beta = 0.9
    gamma = 0.999
    epsilon = 1e-8
    lr = 0.01
    
    m_expected = np.array([0.19, 0.19, 0.19])
    v_expected = np.array([0.01099, 0.01099, 0.01099])
    phi_expected = np.array([0.49723674, 0.49723674, 0.49723674])
    
    phi, m, v = func(iteration_num, gradient, prev_phi, prev_m, prev_v, beta, gamma, epsilon, lr)
    
    assert np.allclose(m, m_expected), f"\033[91mTest case 2 failed. \nExpected m: {m_expected}, but got m: {m}\033[0m"
    assert np.allclose(v, v_expected), f"\033[91mTest case 2 failed. \nExpected v: {v_expected}, but got m: {v}\033[0m"
    assert np.allclose(phi, phi_expected), f"\033[91mTest case 2 failed. \nExpected theta: {phi_expected}, but got theta: {phi}\033[0m"
    
    # Test case 3: Zero gradient should not significantly change weights
    iteration_num = 5
    gradient = np.array([0.0, 0.0, 0.0])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.5, 0.5, 0.5])
    prev_v = np.array([0.25, 0.25, 0.25])
    beta = 0.9
    gamma = 0.999
    epsilon = 1e-8
    lr = 0.001
    
    m_expected = np.array([0.45, 0.45, 0.45])
    v_expected = np.array([0.24975, 0.24975, 0.24975])
    phi_expected = np.array([0.99985133, 0.99985133, 0.99985133])

    phi, m, v = func(iteration_num, gradient, prev_phi, prev_m, prev_v, beta, gamma, epsilon, lr)
    
    assert np.allclose(m, m_expected), f"\033[91mTest case 3 failed. \nExpected m: {m_expected}, but got m: {m}\033[0m"
    assert np.allclose(v, v_expected), f"\033[91mTest case 3 failed. \nExpected v: {v_expected}, but got m: {v}\033[0m"
    assert np.allclose(phi, phi_expected), f"\033[91mTest case 3 failed. \nExpected theta: {phi_expected}, but got theta: {phi}\033[0m"
    
    # Test case 4: High epsilon to observe its effect on numerical stability
    iteration_num = 1
    gradient = np.array([0.1, 0.2, 0.3])
    prev_phi = np.array([1.0, 1.0, 1.0])
    prev_m = np.array([0.0, 0.0, 0.0])
    prev_v = np.array([0.0, 0.0, 0.0])
    beta = 0.9
    gamma = 0.999
    epsilon = 1.0  
    lr = 0.001
    
    m_expected = np.array([0.01, 0.02, 0.03])
    v_expected = np.array([1e-05, 4e-05, 9e-05])
    phi_expected = np.array([0.99995085, 0.99990778, 0.99986974])
    
    phi, m, v = func(iteration_num, gradient, prev_phi, prev_m, prev_v, beta, gamma, epsilon, lr)
    
    assert np.allclose(m, m_expected), f"\033[91mTest case 4 failed. \nExpected m: {m_expected}, but got m: {m}\033[0m"
    assert np.allclose(v, v_expected), f"\033[91mTest case 4 failed. \nExpected v: {v_expected}, but got m: {v}\033[0m"
    assert np.allclose(phi, phi_expected), f"\033[91mTest case 4 failed. \nExpected theta: {phi_expected}, but got theta: {phi}\033[0m"
    
    # Test case 5: Low learning rate should result in small updates
    iteration_num = 1
    gradient = np.array([0.5, 0.5, 0.5])
    prev_phi = np.array([0.5, 0.5, 0.5])
    prev_m = np.array([0.1, 0.1, 0.1])
    prev_v = np.array([0.2, 0.2, 0.2])
    beta = 0.9
    gamma = 0.999
    epsilon = 1e-8
    lr = 1e-5  
    
    m_expected = np.array([0.14, 0.14, 0.14])
    v_expected = np.array([0.20005, 0.20005, 0.20005])
    phi_expected = np.array([0.49999926, 0.49999926, 0.49999926])
    
    phi, m, v = func(iteration_num, gradient, prev_phi, prev_m, prev_v, beta, gamma, epsilon, lr)
    
    assert np.allclose(m, m_expected), f"\033[91mTest case 5 failed. \nExpected m: {m_expected}, but got m: {m}\033[0m"
    assert np.allclose(v, v_expected), f"\033[91mTest case 5 failed. \nExpected v: {v_expected}, but got m: {v}\033[0m"
    assert np.allclose(phi, phi_expected), f"\033[91mTest case 5 failed. \nExpected theta: {phi_expected}, but got theta: {phi}\033[0m"

    print("\033[92mAll tests passed!\033[0m")

def test_auto_reduce(func):
    assert func(0.1, 0.5, 4, 20, 5) == 0.05, f"\033[91mTest case 1 failed. \nExpected alpha: {0.05}, but got alpha: {func(0.1, 0.5, 4, 20, 5)}\033[0m"
    assert func(0.1, 0.5, 8, 20, 5) == 0.025,  f"\033[91mTest case 2 failed. \nExpected alpha: {0.025}, but got alpha: {func(0.1, 0.5, 8, 20, 5)}\033[0m"
    assert func(0.1, 0.5, 0, 20, 5) == 0.1, f"\033[91mTest case 3 failed. \nExpected alpha: {0.1}, but got alpha: {func(0.1, 0.5, 0, 20, 5)}\033[0m"
    assert func(0.1, 0.5, 19, 20, 5) == 0.00625, f"\033[91mTest case 4 failed. \nExpected alpha: {0.0125}, but got alpha: {func(0.1, 0.5, 19, 20, 5)}\033[0m"
    try:
        func(0.1, 0.5, 5, 22, 5)
    except ValueError as e:
        assert str(e) == "max_epochs should be divisible by n_events", f"\033[91mTest case 5 failed. \nExpected error message: max_epochs should be divisible by n_events, but got error message: {func(0.1, 0.5, 5, 22, 5)}\033[0m"
    
    print("\033[92mAll tests passed!\033[0m")

def test_poly_decay(func):
    assert func(0.1, 0.01, 0, 100, 2.0) == 0.1, f"\033[91mTest case 1 failed. \nExpected alpha: {0.1}, but got alpha: {func(0.1, 0.01, 0, 100, 2.0)}\033[0m"
    assert func(0.1, 0.01, 100, 100, 2.0) == 0.009999999999999995,  f"\033[91mTest case 2 failed. \nExpected alpha: {0.009999999999999995}, but got alpha: {func(0.1, 0.01, 100, 100, 2.0)}\033[0m"
    assert func(0.1, 0.01, 50, 100, 1.0) == 0.055, f"\033[91mTest case 3 failed. \nExpected alpha: {0.055}, but got alpha: {func(0.1, 0.01, 50, 100, 1.0)}\033[0m"
    assert func(0.1, 0.2, 5, 50, 0.9) == 0.11258925411794168, f"\033[91mTest case 4 failed. \nExpected alpha: {0.11258925411794168}, but got alpha: {func(0.1, 0.2, 5, 50, 0.9)}\033[0m"

    print("\033[92mAll tests passed!\033[0m")