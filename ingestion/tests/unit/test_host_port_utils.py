# Copyright 2024 Collate
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for details.

"""
Unit tests for host_port_utils.validate_host_port()

Tests both the UI path (builders.py) and 
CLI path (db_utils.py) validation.

Issue: #24348
"""

import pytest
from metadata.utils.host_port_utils import validate_host_port


class TestValidateHostPort:
    """Tests for validate_host_port() function"""

    # ✅ VALID CASES — should pass with no error

    def test_valid_host_and_port(self):
        """Standard format should pass"""
        validate_host_port("localhost:3306")

    def test_valid_ip_and_port(self):
        """IP address format should pass"""
        validate_host_port("192.168.1.100:5432")

    def test_valid_cloud_host(self):
        """Cloud database hostname should pass"""
        validate_host_port(
            "mydb.cluster-xyz.us-east-1.rds.amazonaws.com:3306"
        )

    def test_valid_host_only(self):
        """Host without port should pass"""
        validate_host_port("localhost")

    def test_valid_azure_host(self):
        """Azure database hostname should pass"""
        validate_host_port(
            "myserver.database.windows.net:1433"
        )

    # ❌ INVALID CASES — should raise clear ValueError

    def test_http_prefix_raises_error(self):
        """http:// prefix must raise friendly ValueError"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port("http://localhost:3306")
        error_msg = str(exc_info.value)
        assert "http://" in error_msg
        assert "localhost:3306" in error_msg
        assert "Remove" in error_msg

    def test_https_prefix_raises_error(self):
        """https:// prefix must raise friendly ValueError"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port("https://localhost:3306")
        error_msg = str(exc_info.value)
        assert "https://" in error_msg
        assert "Remove" in error_msg

    def test_http_uppercase_raises_error(self):
        """HTTP:// uppercase must also be caught"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port("HTTP://localhost:3306")
        assert "Remove" in str(exc_info.value)

    def test_https_cloud_host_raises_error(self):
        """https:// with cloud host must raise error"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port(
                "https://mydb.rds.amazonaws.com:3306"
            )
        error_msg = str(exc_info.value)
        assert "https://" in error_msg
        assert "Remove" in error_msg

    def test_empty_string_raises_error(self):
        """Empty hostPort must raise friendly ValueError"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port("")
        assert "empty" in str(exc_info.value).lower()

    def test_error_message_contains_example(self):
        """Error message must contain correct example"""
        with pytest.raises(ValueError) as exc_info:
            validate_host_port("http://localhost:3306")
        assert "localhost:3306" in str(exc_info.value)
