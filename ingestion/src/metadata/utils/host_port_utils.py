# Copyright 2024 Collate
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for details.

"""
Utility functions for validating hostPort connection strings.
"""

import re


def validate_host_port(host_port: str) -> None:
    """
    Validates that the hostPort value is in the correct
    'host:port' format.

    Raises a clear, user-friendly ValueError if the user
    mistakenly provides a full URL like 'http://localhost:3306'
    instead of 'localhost:3306'.

    Args:
        host_port: The hostPort string to validate.

    Raises:
        ValueError: If hostPort contains a URL scheme prefix.
    """
    if not host_port:
        raise ValueError(
            "hostPort cannot be empty. "
            "Please provide a value in 'host:port' format "
            "(e.g. 'localhost:3306')."
        )

    if re.match(r'^https?://', host_port, re.IGNORECASE):
        raise ValueError(
            f"Invalid hostPort format: '{host_port}'. "
            "Please provide the host and port only, "
            "without any URL scheme. "
            "Correct format: 'host:port' "
            "(e.g. 'localhost:3306'). "
            "Remove 'http://' or 'https://' from the value."
        )
