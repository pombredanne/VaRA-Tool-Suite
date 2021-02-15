#!/usr/bin/env python3
"""This module contains custom exceptions."""


class ProcessTerminatedError(Exception):
    """Raised if a process was terminated."""


class ConfigurationLookupError(Exception):
    """Raised if a paper config could not be found."""


class ConfigurationError(Exception):
    """Raised if either a config value is not set or an unexpected config value
    is set."""


class ConfigurationMapConfigIDMissmatch(Exception):
    """Raised if the config ID parsed from a file did not match the actual
    created ID, this happens when IDs are missing in the file."""
