def pytest_configure(config):
    config.addinivalue_line('markers', 'gl: needs an OpenGL context (skips when none can be created)')
