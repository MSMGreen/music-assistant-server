"""Tests for audiobook support in the Jellyfin provider."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any
from unittest import mock

import aiohttp
import pytest
from aiojellyfin import Connection
from aiojellyfin.session import SessionConfiguration
from aiojellyfin.testing import FixtureBuilder
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Audiobook

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


POSITION_S = 3600
POSITION_TICKS = POSITION_S * 10_000_000
NO_CHAPTERS_ID = "00000000000000000000000000000b02"


async def _library_book(mass: MusicAssistant, name: str) -> Audiobook:
    return next(b for b in await mass.music.audiobooks.library_items() if b.name == name)


async def test_resume_position_from_jellyfin(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """The position comes from Jellyfin, with a timestamp so that it wins over local state."""
    provider = _provider(mass, jellyfin_books_provider)
    played, position_ms, timestamp = await provider.get_resume_position(
        BOOK_ID, MediaType.AUDIOBOOK
    )
    assert played is False
    assert position_ms == 3_600_000
    assert timestamp is not None


async def test_resume_position_nothing_in_jellyfin_has_no_timestamp(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """A book Jellyfin holds no state for does not claim to be current."""
    provider = _provider(mass, jellyfin_books_provider)
    assert await provider.get_resume_position(NO_CHAPTERS_ID, MediaType.AUDIOBOOK) == (
        False,
        0,
        None,
    )


async def test_rewinding_in_volume_beats_a_higher_local_position(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """Someone rewound in Volume: playback resumes there, not at the later local position."""
    book = await _library_book(mass, BOOK_NAME)
    # the playlog is per user: without one there would be no local position to compete with
    user = await mass.webserver.auth.create_user(username="listener")
    await mass.music.mark_item_played(
        book,
        fully_played=False,
        seconds_played=9000,
        user_initiated=False,
        userid=user.user_id,
    )
    assert await mass.music.get_resume_position(book, userid=user.user_id) == (False, 3_600_000)


async def test_local_position_kept_when_jellyfin_has_none(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> None:
    """Jellyfin holding nothing must not throw away the place Music Assistant knows."""
    book = await _library_book(mass, NO_CHAPTERS_NAME)
    # the playlog is per user: without one there would be no local position to compete with
    user = await mass.webserver.auth.create_user(username="listener")
    await mass.music.mark_item_played(
        book,
        fully_played=False,
        seconds_played=500,
        user_initiated=False,
        userid=user.user_id,
    )
    assert await mass.music.get_resume_position(book, userid=user.user_id) == (False, 500_000)


@pytest.fixture
async def writes(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig
) -> AsyncGenerator[mock.AsyncMock]:
    """Capture what the provider writes to Jellyfin."""
    provider = _provider(mass, jellyfin_books_provider)
    with mock.patch.object(provider, "_post_user_data", mock.AsyncMock()) as post:
        yield post


async def _on_played(
    mass: MusicAssistant,
    config: ProviderConfig,
    *,
    position: int,
    fully_played: bool = False,
    is_playing: bool = False,
    media_type: MediaType = MediaType.AUDIOBOOK,
) -> None:
    provider = _provider(mass, config)
    await provider.on_played(
        media_type=media_type,
        prov_item_id=BOOK_ID,
        fully_played=fully_played,
        position=position,
        media_item=mock.Mock(),
        is_playing=is_playing,
    )


async def test_on_played_writes_position(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """Progress while playing is written back as ticks, with the time it was set."""
    await _on_played(mass, jellyfin_books_provider, position=4000, is_playing=True)
    writes.assert_awaited_once()
    item_id, data = writes.await_args.args
    assert item_id == BOOK_ID
    assert data["PlaybackPositionTicks"] == 40_000_000_000
    assert data["Played"] is False
    assert data["LastPlayedDate"]


async def test_on_played_finished(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """A finished book is marked played and its position cleared, like Jellyfin itself does."""
    await _on_played(mass, jellyfin_books_provider, position=55648, fully_played=True)
    _, data = writes.await_args.args
    assert data["Played"] is True
    assert data["PlaybackPositionTicks"] == 0


async def test_on_played_last_seconds_still_playing_not_finished(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """The final seconds count as fully played; stopping there must not lose the place."""
    await _on_played(
        mass, jellyfin_books_provider, position=55640, fully_played=True, is_playing=True
    )
    _, data = writes.await_args.args
    assert data["Played"] is False
    assert data["PlaybackPositionTicks"] == 556_400_000_000


async def test_on_played_mark_unplayed(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """Marking a book unplayed in the UI clears it in Jellyfin too."""
    await _on_played(mass, jellyfin_books_provider, position=0)
    _, data = writes.await_args.args
    assert data["Played"] is False
    assert data["PlaybackPositionTicks"] == 0


async def test_on_played_never_writes_zero_while_playing(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """A zero position during playback is noise, and would wipe the household's place."""
    await _on_played(mass, jellyfin_books_provider, position=0, is_playing=True)
    writes.assert_not_awaited()


async def test_on_played_ignores_tracks(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """Music playback is not written back to Jellyfin; that behaviour is unchanged."""
    await _on_played(
        mass, jellyfin_books_provider, position=100, is_playing=True, media_type=MediaType.TRACK
    )
    writes.assert_not_awaited()


async def test_on_played_does_not_clobber_newer_position(
    mass: MusicAssistant,
    jellyfin_books_provider: ProviderConfig,
    fixture_builder: FixtureBuilder,
    writes: mock.AsyncMock,
) -> None:
    """Volume moved on since playback started here: a late report must not undo that."""
    provider = _provider(mass, jellyfin_books_provider)
    await provider.get_resume_position(BOOK_ID, MediaType.AUDIOBOOK)
    fixture_builder.objects[BOOK_ID]["UserData"]["PlaybackPositionTicks"] = 90_000_000_000
    await _on_played(mass, jellyfin_books_provider, position=3700)
    writes.assert_not_awaited()


async def test_on_played_continues_after_own_writes(
    mass: MusicAssistant,
    jellyfin_books_provider: ProviderConfig,
    fixture_builder: FixtureBuilder,
    writes: mock.AsyncMock,
) -> None:
    """The position moving because of our own write is not mistaken for Volume moving it."""
    provider = _provider(mass, jellyfin_books_provider)
    await provider.get_resume_position(BOOK_ID, MediaType.AUDIOBOOK)
    await _on_played(mass, jellyfin_books_provider, position=3630, is_playing=True)
    fixture_builder.objects[BOOK_ID]["UserData"]["PlaybackPositionTicks"] = 36_300_000_000
    await _on_played(mass, jellyfin_books_provider, position=3660, is_playing=True)
    assert writes.await_count == 2


async def test_on_played_survives_jellyfin_being_down(
    mass: MusicAssistant, jellyfin_books_provider: ProviderConfig, writes: mock.AsyncMock
) -> None:
    """A failed write is logged, not raised into the playback reporting task."""
    writes.side_effect = aiohttp.ClientError("down")
    await _on_played(mass, jellyfin_books_provider, position=4000, is_playing=True)
    writes.assert_awaited_once()
