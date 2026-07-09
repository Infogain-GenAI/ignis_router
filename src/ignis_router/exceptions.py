"""Custom exceptions for the ignis_router library."""


class IgnisRouterError(Exception):
    """Base exception for all ignis_router errors."""

    pass


class RoutingError(IgnisRouterError):
    """Raised when routing fails to select a model."""

    pass


class IntentDetectionError(IgnisRouterError):
    """Raised when intent detection fails."""

    pass


class ModelNotAvailableError(IgnisRouterError):
    """Raised when the selected model is not available or not registered."""

    pass


class ConfigurationError(IgnisRouterError):
    """Raised when there is a configuration issue."""

    pass
