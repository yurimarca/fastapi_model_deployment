# Tests Directory

This folder contains the test suite for the FastAPI model deployment project. The tests are managed using `pytest`.

## Structure

- `conftest.py`: Contains shared fixtures and configurations for the test suite.
- Other test files: Individual test cases for various components of the project.

## Running Tests

To execute the tests with all useful flags for CI environments like GitHub Actions, run the following command from this directory:

```bash
pytest --maxfail=1 --disable-warnings -v
```

- `--maxfail=1`: Stops the test suite after the first failure, saving time in CI pipelines.
- `--disable-warnings`: Suppresses warning messages for cleaner output.
- `-v`: Enables verbose mode for detailed test results.

Ensure all dependencies are installed before running the tests.