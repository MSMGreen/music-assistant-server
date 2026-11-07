"""Test the tracks of a Deezer playlist: when each was added and how they are numbered."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientError
from deezer_python_gql import GraphQLClientHttpError
from music_assistant_models.media_items import ProviderMapping, Track

from music_assistant.providers.deezer import browse
from music_assistant.providers.deezer.media import DeezerMediaManager
from music_assistant.providers.deezer.provider import DeezerProvider
from tests.common import use_real_create_task


def _edge(track_id: str | None, added_at: str | None) -> SimpleNamespace:
    """Return a playlist track edge as the GraphQL client parses it."""
    node = SimpleNamespace(id=track_id) if track_id else None
    return SimpleNamespace(node=node, added_at=added_at)


def _page(edges: list[SimpleNamespace], next_cursor: str | None = None) -> SimpleNamespace:
    """Return one page of a playlist's tracks."""
    page_info = SimpleNamespace(has_next_page=next_cursor is not None, end_cursor=next_cursor)
    return SimpleNamespace(tracks=SimpleNamespace(edges=edges, page_info=page_info))


def _parse_track(provider: DeezerProvider, node: Any, position: int = 0) -> Track:
    """Stand in for parse_track: a minimal track for the node."""
    return Track(
        item_id=node.id,
        provider=provider.instance_id,
        name=node.id,
        position=position,
        provider_mappings={
            ProviderMapping(
                item_id=node.id,
                provider_domain="deezer",
                provider_instance=provider.instance_id,
            )
        },
    )


def _serve_pages(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch, pages: list[SimpleNamespace]
) -> AsyncMock:
    """Make the provider fetch these playlist pages, bypassing the cache."""
    monkeypatch.setattr(browse, "parse_track", _parse_track)
    provider.mass.cache.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    provider.mass.cache.set = AsyncMock()  # type: ignore[method-assign]
    use_real_create_task(provider.mass)
    remaining = iter(pages)

    async def get_playlist(**kwargs: Any) -> SimpleNamespace:
        # a request for one track only asks when the playlist last changed
        if kwargs.get("tracks_first") == 1:
            return SimpleNamespace(last_modification_date="2024-06-01T10:00:00.000Z")
        return next(remaining)

    get = AsyncMock(side_effect=get_playlist)
    provider.gql_client.get_playlist = get  # type: ignore[method-assign]
    return get


@pytest.mark.asyncio
async def test_playlist_tracks_carry_date_added(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each track gets the time it was added to the playlist, across pages."""
    _serve_pages(
        provider,
        monkeypatch,
        [
            _page(
                [_edge("1", "2024-03-01T12:00:00.000Z"), _edge(None, "2024-03-01T12:00:00.000Z")],
                next_cursor="c1",
            ),
            _page([_edge("3", "2024-05-10T08:30:00.000Z"), _edge("4", None)]),
        ],
    )

    tracks = await provider.browse_manager.get_playlist_tracks("123")

    assert [(t.item_id, t.position) for t in tracks] == [("1", 1), ("3", 2), ("4", 3)]
    assert [t.date_added for t in tracks] == [
        datetime(2024, 3, 1, 12, 0, 0, tzinfo=UTC),
        datetime(2024, 5, 10, 8, 30, 0, tzinfo=UTC),
        None,
    ]


@pytest.mark.asyncio
async def test_move_to_top_past_an_unavailable_track(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable first track leaves no gap that a block move could pick as its target."""
    page = _page([_edge(None, None), _edge("1", None), _edge("2", None)])
    # the playlist is read once to plan the block move and once for the move itself
    _serve_pages(provider, monkeypatch, [page, page])
    provider.mass.cache.delete = AsyncMock()  # type: ignore[method-assign]
    provider.media_manager = DeezerMediaManager(provider)
    move = AsyncMock(return_value=SimpleNamespace(move_status=True))
    provider.gql_client.move_track_in_playlist = move  # type: ignore[method-assign]

    await provider.move_playlist_tracks("123", [2], 1)

    move.assert_awaited_once_with(playlist_id="123", track_id="2", after_track_id=None)


@pytest.mark.asyncio
async def test_unchanged_playlist_comes_from_the_cache(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A playlist that did not change since it was cached is not fetched again."""
    get = _serve_pages(provider, monkeypatch, [])
    cached = [_parse_track(provider, SimpleNamespace(id="1"), position=1)]
    cache_get = AsyncMock(return_value=cached)
    provider.mass.cache.get = cache_get  # type: ignore[method-assign]

    assert await provider.browse_manager.get_playlist_tracks("123") == cached
    cache_get.assert_awaited_once_with(
        "_get_regular_playlist_tracks.123",
        provider=provider.instance_id,
        checksum="2024-06-01T10:00:00.000Z",
        base_class=Track,
    )
    get.assert_awaited_once_with(playlist_id="123", tracks_first=1)


@pytest.mark.asyncio
async def test_changed_playlist_is_fetched_and_cached(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed playlist is fetched again and stored with its new modification date."""
    _serve_pages(provider, monkeypatch, [_page([_edge("1", None)])])
    cache_set = AsyncMock()
    provider.mass.cache.set = cache_set  # type: ignore[method-assign]

    tracks = await provider.browse_manager.get_playlist_tracks("123")

    assert [t.item_id for t in tracks] == ["1"]
    cache_set.assert_awaited_once_with(
        "_get_regular_playlist_tracks.123",
        tracks,
        expiration=3600 * 3,
        provider=provider.instance_id,
        checksum="2024-06-01T10:00:00.000Z",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [GraphQLClientHttpError(status_code=502, response=Mock()), ClientError(), TimeoutError()],
)
async def test_last_tracks_while_deezer_does_not_answer(
    provider: DeezerProvider, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Without an answer from Deezer the last cached tracks are used, whatever their date."""
    _serve_pages(provider, monkeypatch, [])
    provider.gql_client.get_playlist = AsyncMock(side_effect=error)  # type: ignore[method-assign]
    cached = [_parse_track(provider, SimpleNamespace(id="1"), position=1)]
    cache_get = AsyncMock(return_value=cached)
    provider.mass.cache.get = cache_get  # type: ignore[method-assign]

    assert await provider.browse_manager.get_playlist_tracks("123") == cached
    cache_get.assert_awaited_once_with(
        "_get_regular_playlist_tracks.123", provider=provider.instance_id, base_class=Track
    )
