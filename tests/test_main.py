# MIT License
# Copyright (c) 2025 ChaseDDevelopment
# See LICENSE file for full license information.
"""Tests for the main module."""

from unittest.mock import patch
import subprocess
import sys
import main


def test_main():
    """Test the main function."""
    with patch("builtins.print") as mock_print:
        main.main()
        mock_print.assert_called_once_with("Hello from morpheus!")


def test_main_as_script():
    """Test running main as a script."""
    # Run the script as a subprocess to test the if __name__ == "__main__" block
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    assert result.stdout.strip() == "Hello from morpheus!"
    assert result.returncode == 0
