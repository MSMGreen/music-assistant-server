"""Tests for audiobook support in the Jellyfin provider."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
from aiojellyfin import Connection
from aiojellyfin.session import SessionConfiguration
from aiojellyfin.testing import FixtureBuilder
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError

from music_assistant.mass import MusicAssistant
from music_assistant.providers.jellyfin import JellyfinProvider
from tests.common import get_fixtures_dir, wait_for_sync_completion

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig

BOOK_ID = "c022338cd4fa4f51de8ae471c7e978d8"
BOOK_NAME = "Discount Dan - Discount Dan’s Backroom Bargains, Book 1"
NO_CHAPTERS_NAME = "A Book Without Chapters"
BOOKS_FOLDER = "books-folder"
BOOKS_FOLDER_ITEM = {"Id": BOOKS_FOLDER, "Name": "Books", "CollectionType": "books"}


async def _build_fixture() -> FixtureBuilder:
    """Return a fixture with the music fixtures plus the audiobook fixtures."""
    f = FixtureBuilder()
    for kind in ("artists", "albums", "tracks"):
        async for _, data in get_fixtures_dir(kind, "jellyfin"):
            f.add_json_bytes(data)
    async for _, data in get_fixtures_dir("audiobooks", "jellyfin"):
        f.add_json_bytes(data)
    for record_id, record in f.objects.items():
        if record.get("Type") in ("AudioBook", "Book"):
            f.objects_parents[record_id] = {BOOKS_FOLDER}
    return f


def _with_books_folder(
    authenticate: Callable[[SessionConfiguration, str, str], Awaitable[Connection]],
) -> Callable[[SessionConfiguration, str, str], Awaitable[Connection]]:
    """Wrap a fixture login so the connection also reports a books library."""

    async def authenticate_by_name(
        session_config: SessionConfiguration, username: str, password: str
    ) -> Connection:
        connection = await authenticate(session_config, username, password)
        music_folders = connection.get_media_folders

        async def get_media_folders(fields: str | None = None) -> Any:
            folders = await music_folders(fields)
            return {**folders, "Items": [*folders["Items"], BOOKS_FOLDER_ITEM]}

        connection.get_media_folders = get_media_folders  # type: ignore[method-assign]
        return connection

    return authenticate_by_name


@pytest.fixture
async def fixture_builder() -> FixtureBuilder:
    """Return the shared test data; tests may edit it before or after sync."""
    return await _build_fixture()


@pytest.fixture
async def jellyfin_books_provider(
    mass: MusicAssistant, fixture_builder: FixtureBuilder
) -> AsyncGenerator[ProviderConfig]:
    """Add a Jellyfin provider whose server has a music and a books library."""
    authenticate_by_name = _with_books_folder(fixture_builder.to_authenticate_by_name())
    with mock.patch(
        "music_assistant.providers.jellyfin.authenticate_by_name", authenticate_by_name
    ):
        async with wait_for_sync_completion(mass):
            config = await mass.config._create_provider_instance(
                "jellyfin",
                {},
                setup_data=mass.config._encrypt_values(
                    {"url": "http://localhost", "username": "username", "password": "password"}
                ),
            )
            await mass.music.start_sync()
        yield config


def _provider(mass: MusicAssistant, config: ProviderConfig) -> JellyfinProvider:
    provider = mass.get_provider(config.instance_id)
    assert isinstance(provider, JellyfinProvider)
    return provider


async def test_audiobooks_synced(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """Single-file audiobooks are in the library; ebooks and multi-file parts are not."""
    books = await mass.music.audiobooks.library_items()
    assert {b.name for b in books} == {BOOK_NAME, NO_CHAPTERS_NAME}


async def test_multifile_book_skipped_with_warning(
    mass: MusicAssistant,
    fixture_builder: FixtureBuilder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A book made of several files is skipped with one warning naming its folder."""
    caplog.set_level(logging.WARNING)
    authenticate_by_name = _with_books_folder(fixture_builder.to_authenticate_by_name())
    with mock.patch(
        "music_assistant.providers.jellyfin.authenticate_by_name", authenticate_by_name
    ):
        async with wait_for_sync_completion(mass):
            await mass.config._create_provider_instance(
                "jellyfin",
                {},
                setup_data=mass.config._encrypt_values(
                    {"url": "http://localhost", "username": "username", "password": "password"}
                ),
            )
            await mass.music.start_sync()
    warnings = [r for r in caplog.records if "multi-parent" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


async def test_music_still_synced(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """Adding books does not disturb the music library."""
    tracks = await mass.music.tracks.library_items(search="where the bands are")
    assert tracks


async def test_get_audiobook(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """A single audiobook can be fetched by id; an unknown id is not found."""
    provider = _provider(mass, jellyfin_books_provider)
    book = await provider.get_audiobook(BOOK_ID)
    assert book.name == BOOK_NAME
    with pytest.raises(MediaNotFoundError):
        await provider.get_audiobook("ffffffffffffffffffffffffffffffff")


async def test_audiobook_stream_details(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """An audiobook streams from Jellyfin, is seekable and knows its length."""
    provider = _provider(mass, jellyfin_books_provider)
    stream = await provider.get_stream_details(BOOK_ID, MediaType.AUDIOBOOK)
    assert stream.duration == 55648
    assert stream.can_seek
    assert stream.allow_seek
    assert stream.media_type == MediaType.AUDIOBOOK
    assert BOOK_ID in stream.path
    assert "apiKey=" in stream.path


async def test_track_stream_details_unchanged(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """Streaming a track still works and is still a track."""
    provider = _provider(mass, jellyfin_books_provider)
    track_id = "da9c458e425584680765ddc3a89cbc0c"
    stream = await provider.get_stream_details(track_id, MediaType.TRACK)
    assert stream.media_type == MediaType.TRACK
    assert track_id in stream.path
