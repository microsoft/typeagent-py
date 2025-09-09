# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import pytest
from pytest_mock import MockerFixture

from typeagent.aitools.auth import AzureTokenProvider, get_shared_token_provider


@pytest.fixture
def mock_azure_token_provider(mocker: MockerFixture) -> AzureTokenProvider:
    """Fixture to create a mocked AzureTokenProvider."""
    mock_credential = mocker.patch(
        "typeagent.aitools.auth.DefaultAzureCredential"
    ).return_value
    provider = AzureTokenProvider()
    provider.credential = mock_credential
    return provider


def test_get_token_with_refresh(
    mock_azure_token_provider: AzureTokenProvider, mocker: MockerFixture
):
    """Test that get_token refreshes the token when needed."""
    mocker.patch.object(mock_azure_token_provider, "needs_refresh", return_value=True)
    mock_refresh_token = mocker.patch.object(
        mock_azure_token_provider, "refresh_token", return_value="new_token"
    )

    token = mock_azure_token_provider.get_token()

    mock_azure_token_provider.needs_refresh.assert_called_once()  # type: ignore
    mock_refresh_token.assert_called_once()
    assert token == "new_token"


def test_get_token_without_refresh(
    mock_azure_token_provider: AzureTokenProvider, mocker: MockerFixture
):
    """Test that get_token uses the cached token when refresh is not needed."""
    mocker.patch.object(mock_azure_token_provider, "needs_refresh", return_value=False)
    mock_azure_token_provider.access_token = mocker.MagicMock(token="cached_token")

    token = mock_azure_token_provider.get_token()

    mock_azure_token_provider.needs_refresh.assert_called_once()  # type: ignore
    assert token == "cached_token"


def test_refresh_token(
    mock_azure_token_provider: AzureTokenProvider, mocker: MockerFixture
):
    """Test that refresh_token retrieves a new token."""
    mock_get_token = mocker.patch.object(
        mock_azure_token_provider.credential,
        "get_token",
        return_value=mocker.MagicMock(
            token="new_token", expires_on=int(time.time()) + 3600
        ),
    )

    token = mock_azure_token_provider.refresh_token()

    mock_get_token.assert_called_once_with(
        "https://cognitiveservices.azure.com/.default"
    )
    assert token == "new_token"
    assert mock_azure_token_provider.access_token is not None
    assert mock_azure_token_provider.access_token.token == "new_token"


def test_needs_refresh_no_token(mock_azure_token_provider: AzureTokenProvider):
    """Test that needs_refresh returns True when no token is present."""
    mock_azure_token_provider.access_token = None
    assert mock_azure_token_provider.needs_refresh() is True


def test_needs_refresh_expired_token(
    mock_azure_token_provider: AzureTokenProvider, mocker: MockerFixture
):
    """Test that needs_refresh returns True when the token is about to expire."""
    mock_azure_token_provider.access_token = mocker.MagicMock(
        expires_on=int(time.time()) + 100
    )
    assert mock_azure_token_provider.needs_refresh() is True


def test_needs_refresh_valid_token(
    mock_azure_token_provider: AzureTokenProvider, mocker: MockerFixture
):
    """Test that needs_refresh returns False when the token is valid."""
    mock_azure_token_provider.access_token = mocker.MagicMock(
        expires_on=int(time.time()) + 3600
    )
    assert mock_azure_token_provider.needs_refresh() is False


def test_get_shared_token_provider():
    """Test that get_shared_token_provider returns a singleton instance."""
    provider1 = get_shared_token_provider()
    provider2 = get_shared_token_provider()
    assert provider1 is provider2
