"""Jellyfin support for MusicAssistant."""

from __future__ import annotations

import hashlib
import socket
from asyncio import TaskGroup
from collections import Counter
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from aiohttp import ClientError
from aiojellyfin import MediaLibrary as JellyMediaLibrary
from aiojellyfin import NotFound, authenticate_by_name
from aiojellyfin.session import NotFound as SessionNotFound
from aiojellyfin.session import SessionConfiguration
from music_assistant_models.enums import MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import LoginFailed, MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    Audiobook,
    MediaItemType,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import UNKNOWN_ARTIST, UNKNOWN_ARTIST_ID_MBID
from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.datetime import utc
from music_assistant.mass import MusicAssistant
from music_assistant.models import ProviderInstanceType
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.jellyfin.parsers import (
    audio_format,
    parse_album,
    parse_artist,
    parse_audiobook,
    parse_playlist,
    parse_track,
)

from .const import (
    ALBUM_FIELDS,
    ARTIST_FIELDS,
    AUDIOBOOK_FIELDS,
    AUDIOBOOK_PAGE_SIZE,
    AUDIOBOOK_PROGRESS_POLL_SECONDS,
    CACHE_EXPIRATION_SEEN_PROGRESS,
    CACHE_KEY_KNOWN_TICKS,
    CACHE_KEY_SEEN_PROGRESS,
    COLLECTION_TYPE_BOOKS,
    COLLECTION_TYPE_MUSIC,
    COLLECTION_TYPE_PLAYLISTS,
    ITEM_KEY_COLLECTION_TYPE,
    ITEM_KEY_ID,
    ITEM_KEY_MEDIA_STREAMS,
    ITEM_KEY_MEDIA_TYPE,
    ITEM_KEY_NAME,
    ITEM_KEY_PARENT_ID,
    ITEM_KEY_RUNTIME_TICKS,
    ITEM_KEY_TYPE,
    ITEM_KEY_USER_DATA,
    ITEM_TYPE_AUDIOBOOK,
    MEDIA_TYPE_AUDIO,
    SUPPORTED_CONTAINER_FORMATS,
    TICKS_PER_MS,
    TICKS_PER_SECOND,
    TRACK_FIELDS,
    USER_APP_NAME,
    USER_DATA_KEY_LAST_PLAYED,
    USER_DATA_KEY_PLAYED,
    USER_DATA_KEY_POSITION_TICKS,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

CONF_URL = "url"
CONF_USERNAME = "username"
CONF_PASSWORD = "password"
CONF_VERIFY_SSL = "verify_ssl"
SUPPORTED_FEATURES = {
    ProviderFeature.LIBRARY_ARTISTS,
    ProviderFeature.LIBRARY_ALBUMS,
    ProviderFeature.LIBRARY_TRACKS,
    ProviderFeature.LIBRARY_PLAYLISTS,
    ProviderFeature.LIBRARY_AUDIOBOOKS,
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.SIMILAR_TRACKS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return JellyfinProvider(mass, manifest, config, SUPPORTED_FEATURES)


class JellyfinProvider(MusicProvider):
    """Provider for a jellyfin music library."""

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to setup this provider."""
        return ()

    async def handle_async_init(self) -> None:
        """Initialize provider(instance) with given configuration."""
        # per audiobook, the position in Jellyfin we last read at play start or wrote ourselves.
        # Kept over restarts, because a parked queue survives them and resumes without asking
        # Jellyfin: its reports must still not overwrite a place another client moved on
        self._known_ticks: dict[str, int] = dict(
            await self.mass.cache.get(CACHE_KEY_KNOWN_TICKS, provider=self.instance_id, default={})
        )
        # per audiobook, the [ticks, played] last reflected in Music Assistant's playlog. Kept
        # over restarts: pushing every book again would scramble which one was played last
        self._seen_progress: dict[str, list[Any]] = dict(
            await self.mass.cache.get(
                CACHE_KEY_SEEN_PROGRESS, provider=self.instance_id, default={}
            )
        )
        username = str(self.get_setup_value(CONF_USERNAME))

        # Device ID should be stable between reboots
        # Otherwise every time the provider starts we "leak" a new device
        # entry in the Jellyfin backend, which creates devices and entities
        # in HA if they also use the Jellyfin integration there.

        # We follow a suggestion a Jellyfin dev gave to HA and use an ID
        # that is stable even if provider is removed and re-added.
        # They said mix in username in case the same device/app has 2
        # connections to the same servers

        # Neither of these are secrets (username is handed over to mint a
        # token and server_id is used in zeroconf) but hash them anyway as its meant
        # to be an opaque identifier

        device_id = hashlib.sha256(f"{self.mass.server_id}+{username}".encode()).hexdigest()
        verify_ssl = bool(self.get_setup_value(CONF_VERIFY_SSL))
        http_session = self.mass.http_session if verify_ssl else self.mass.http_session_no_ssl

        session_config = SessionConfiguration(
            session=http_session,
            url=str(self.get_setup_value(CONF_URL)),
            verify_ssl=verify_ssl,
            app_name=USER_APP_NAME,
            app_version=self.mass.version,
            device_name=socket.gethostname(),
            device_id=device_id,
        )

        try:
            self._client = await authenticate_by_name(
                session_config,
                username,
                str(self.get_setup_value(CONF_PASSWORD) or ""),
            )
        except Exception as err:
            raise LoginFailed(f"Authentication failed: {err}") from err

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return False

    async def resolve_image(self, path: str) -> str | bytes:
        """
        Return an accessible Jellyfin artwork URL.

        :param path: Artwork URL or provider path to resolve.
        """
        return _normalize_jellyfin_media_url(path)

    async def loaded_in_mass(self) -> None:
        """Start picking up audiobook progress from other Jellyfin clients."""
        self._schedule_audiobook_progress_poll()

    async def unload(self, is_removed: bool = False) -> None:
        """
        Stop picking up audiobook progress.

        :param is_removed: Whether the provider is removed from the configuration.
        """
        self.mass.cancel_timer(self._progress_poll_task_id)

    async def get_library_audiobooks(self) -> AsyncGenerator[Audiobook]:
        """Retrieve all single-file audiobooks from the Jellyfin books libraries."""
        # A book made of several files shows up in Jellyfin as several AudioBook items sharing
        # a parent folder. Music Assistant could only play those as unrelated books, each with
        # its own resume point, so they are skipped.
        for library in await self._get_book_libraries():
            books = [book async for book in self._iter_raw_audiobooks(library[ITEM_KEY_ID])]
            per_parent = Counter(book.get(ITEM_KEY_PARENT_ID) for book in books)
            warned: set[str] = set()
            for book in books:
                parent = book.get(ITEM_KEY_PARENT_ID)
                if parent and per_parent[parent] > 1:
                    if parent not in warned:
                        warned.add(parent)
                        self.logger.warning(
                            "Skipping multi-file audiobook in folder %s: %s files",
                            parent,
                            per_parent[parent],
                        )
                    continue
                try:
                    parsed = parse_audiobook(self.instance_id, self._client, book)
                except (KeyError, ValueError) as err:
                    self.report_skipped_sync_item(MediaType.AUDIOBOOK, book.get(ITEM_KEY_ID), err)
                    continue
                # library sync writes any resume state it is given into the playlog, as played
                # now. That would bump books with nobody listening, so only the poll reports it
                parsed.resume_position_ms = None
                parsed.fully_played = None
                yield parsed

    async def get_audiobook(self, prov_audiobook_id: str) -> Audiobook:
        """Get full audiobook details by id."""
        book = await self._get_raw_audiobook(prov_audiobook_id)
        return parse_audiobook(self.instance_id, self._client, book)

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, datetime | None]:
        """
        Get the resume point of an audiobook from Jellyfin.

        :param item_id: Id of the audiobook.
        :param media_type: The media type of the item.
        """
        if media_type != MediaType.AUDIOBOOK:
            raise NotImplementedError
        book = await self._get_raw_audiobook(item_id)
        user_data = book.get(ITEM_KEY_USER_DATA) or {}
        ticks = int(user_data.get(USER_DATA_KEY_POSITION_TICKS) or 0)
        played = bool(user_data.get(USER_DATA_KEY_PLAYED, False))
        await self._remember_known_ticks(item_id, ticks)
        # Jellyfin holds the one shared place in a book, which other clients move too, mostly
        # without recording when. Without a newer timestamp Music Assistant takes the higher of
        # its own position and this one, which would ignore a rewind in another app, so a position
        # Jellyfin holds is reported as current. When it holds nothing there is no timestamp, so
        # a place only Music Assistant knows is kept.
        if not ticks and not played:
            return False, 0, None
        return played, ticks // TICKS_PER_MS, utc()

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Write the progress of an audiobook back to Jellyfin.

        :param media_type: The media type of the played item.
        :param prov_item_id: Id of the played item.
        :param fully_played: Whether the item was played to its end.
        :param position: Last known position in seconds.
        :param media_item: The played item.
        :param is_playing: Whether the item is still playing.
        """
        if media_type != MediaType.AUDIOBOOK:
            return
        # the last seconds of a book count as fully played while they are still playing
        finished = fully_played and not is_playing
        marked_unplayed = not fully_played and not position and not is_playing
        if not finished and not marked_unplayed and not position:
            # a zero position during playback means nothing, and would wipe the shared place
            return
        ticks = 0 if finished or marked_unplayed else int(position * TICKS_PER_SECOND)
        try:
            if not finished and not marked_unplayed and await self._moved_elsewhere(prov_item_id):
                self.logger.warning(
                    "Not saving progress of %s: another client moved it since playback started",
                    prov_item_id,
                )
                return
            await self._post_user_data(
                prov_item_id,
                {
                    USER_DATA_KEY_POSITION_TICKS: ticks,
                    USER_DATA_KEY_PLAYED: finished,
                    USER_DATA_KEY_LAST_PLAYED: utc().isoformat(),
                },
            )
        except (ClientError, TimeoutError, MediaNotFoundError) as err:
            self.logger.warning("Could not save progress of %s: %s", prov_item_id, err)
            return
        await self._remember_known_ticks(prov_item_id, ticks)
        # already in the playlog, since that is where this report came from
        self._seen_progress[prov_item_id] = [ticks, finished]
        if not is_playing:
            await self._store_seen_progress()

    async def _search_track(self, search_query: str, limit: int) -> list[Track]:
        resultset = (
            await self._client.tracks.search_term(search_query)
            .limit(limit)
            .enable_userdata()
            .fields(*TRACK_FIELDS)
            .request()
        )
        tracks = []
        for item in resultset["Items"]:
            tracks.append(parse_track(self.logger, self.instance_id, self._client, item))
        return tracks

    async def _search_album(self, search_query: str, limit: int) -> list[Album]:
        # an "Artist - Album" style query: search on the album part only
        albumname = search_query.split(" - ", 1)[1] if " - " in search_query else search_query
        resultset = (
            await self._client.albums.search_term(albumname)
            .limit(limit)
            .enable_userdata()
            .fields(*ALBUM_FIELDS)
            .request()
        )
        albums = []
        for item in resultset["Items"]:
            albums.append(parse_album(self.logger, self.instance_id, self._client, item))
        return albums

    async def _search_artist(self, search_query: str, limit: int) -> list[Artist]:
        resultset = (
            await self._client.artists.search_term(search_query)
            .limit(limit)
            .enable_userdata()
            .fields(*ARTIST_FIELDS)
            .request()
        )
        artists = []
        for item in resultset["Items"]:
            artists.append(parse_artist(self.logger, self.instance_id, self._client, item))
        return artists

    async def _search_playlist(self, search_query: str, limit: int) -> list[Playlist]:
        resultset = (
            await self._client.playlists.search_term(search_query)
            .limit(limit)
            .enable_userdata()
            .request()
        )
        playlists = []
        for item in resultset["Items"]:
            playlists.append(parse_playlist(self.instance_id, self._client, item))
        return playlists

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 20,
    ) -> SearchResults:
        """
        Perform search on the Jellyfin library.

        :param search_query: Search query.
        :param media_types: A list of media_types to include. All types if None.
        :param limit: Number of items to return in the search (per type).
        """
        artists = None
        albums = None
        tracks = None
        playlists = None

        async with TaskGroup() as tg:
            if MediaType.ARTIST in media_types:
                artists = tg.create_task(self._search_artist(search_query, limit))
            if MediaType.ALBUM in media_types:
                albums = tg.create_task(self._search_album(search_query, limit))
            if MediaType.TRACK in media_types:
                tracks = tg.create_task(self._search_track(search_query, limit))
            if MediaType.PLAYLIST in media_types:
                playlists = tg.create_task(self._search_playlist(search_query, limit))

        search_results = SearchResults()

        if artists:
            search_results.artists = artists.result()
        if albums:
            search_results.albums = albums.result()
        if tracks:
            search_results.tracks = tracks.result()
        if playlists:
            search_results.playlists = playlists.result()

        return search_results

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Retrieve all library artists from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            stream = (
                self._client.artists.parent(jellyfin_library[ITEM_KEY_ID])
                .enable_userdata()
                .fields(*ARTIST_FIELDS)
                .stream(100)
            )
            async for artist in stream:
                yield parse_artist(self.logger, self.instance_id, self._client, artist)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Retrieve all library albums from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            stream = (
                self._client.albums.parent(jellyfin_library[ITEM_KEY_ID])
                .enable_userdata()
                .fields(*ALBUM_FIELDS)
                .stream(100)
            )
            async for album in stream:
                yield parse_album(self.logger, self.instance_id, self._client, album)

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Retrieve library tracks from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            stream = (
                self._client.tracks.parent(jellyfin_library[ITEM_KEY_ID])
                .enable_userdata()
                .fields(*TRACK_FIELDS)
                .stream(100)
            )
            async for track in stream:
                if not len(track[ITEM_KEY_MEDIA_STREAMS]):
                    self.logger.warning(
                        "Invalid track %s: Does not have any media streams", track[ITEM_KEY_NAME]
                    )
                    continue
                yield parse_track(self.logger, self.instance_id, self._client, track)

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Retrieve all library playlists from the provider."""
        playlist_libraries = await self._get_playlists()
        for playlist_library in playlist_libraries:
            stream = (
                self._client.playlists.parent(playlist_library[ITEM_KEY_ID])
                .enable_userdata()
                .stream(100)
            )
            async for playlist in stream:
                if ITEM_KEY_MEDIA_TYPE in playlist:  # Only jellyfin has this property
                    if playlist[ITEM_KEY_MEDIA_TYPE] == MEDIA_TYPE_AUDIO:
                        yield parse_playlist(self.instance_id, self._client, playlist)
                else:  # emby playlists are only audio type
                    yield parse_playlist(self.instance_id, self._client, playlist)

    async def get_album(self, prov_album_id: str) -> Album:
        """Get full album details by id."""
        try:
            album = await self._client.get_album(prov_album_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_album_id} not found")
        return parse_album(self.logger, self.instance_id, self._client, album)

    @use_cache(3600)  # Cache for 1 hour
    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Get album tracks for given album id."""
        jellyfin_album_tracks = (
            await self._client.tracks.parent(prov_album_id)
            .enable_userdata()
            .fields(*TRACK_FIELDS)
            .request()
        )
        return [
            parse_track(self.logger, self.instance_id, self._client, jellyfin_album_track)
            for jellyfin_album_track in jellyfin_album_tracks["Items"]
        ]

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        if prov_artist_id == UNKNOWN_ARTIST:
            artist = Artist(
                item_id=UNKNOWN_ARTIST,
                name=UNKNOWN_ARTIST,
                provider=self.instance_id,
                provider_mappings={
                    ProviderMapping(
                        item_id=UNKNOWN_ARTIST,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                },
            )
            artist.mbid = UNKNOWN_ARTIST_ID_MBID
            return artist

        try:
            jellyfin_artist = await self._client.get_artist(prov_artist_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_artist_id} not found")
        return parse_artist(self.logger, self.instance_id, self._client, jellyfin_artist)

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        try:
            track = await self._client.get_track(prov_track_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_track_id} not found")
        return parse_track(self.logger, self.instance_id, self._client, track)

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id."""
        try:
            playlist = await self._client.get_playlist(prov_playlist_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_playlist_id} not found")
        return parse_playlist(self.instance_id, self._client, playlist)

    @use_cache(3600)  # Cache for 1 hour
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Get playlist tracks."""
        result: list[Track] = []
        playlist_items = (
            await self._client.tracks.in_playlist(prov_playlist_id)
            .enable_userdata()
            .fields(*TRACK_FIELDS)
            .limit(100)
            .start_index(page * 100)
            .request()
        )
        for index, jellyfin_track in enumerate(playlist_items["Items"], 1):
            pos = (page * 100) + index
            try:
                if track := parse_track(
                    self.logger, self.instance_id, self._client, jellyfin_track
                ):
                    track.position = pos
                    result.append(track)
            except (KeyError, ValueError) as err:
                self.logger.error(
                    "Skipping track %s: %s", jellyfin_track.get(ITEM_KEY_NAME, index), str(err)
                )
        return result

    @use_cache(3600)  # Cache for 1 hour
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get a list of albums for the given artist."""
        albums = (
            await self._client.albums.parent(prov_artist_id)
            .fields(*ALBUM_FIELDS)
            .enable_userdata()
            .request()
        )
        return [
            parse_album(self.logger, self.instance_id, self._client, album)
            for album in albums["Items"]
        ]

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given track or audiobook when it will be streamed."""
        jellyfin_track: Any
        if media_type == MediaType.AUDIOBOOK:
            jellyfin_track = await self._get_raw_audiobook(item_id)
        else:
            try:
                jellyfin_track = await self._client.get_track(item_id)
            except NotFound:
                raise MediaNotFoundError(f"Item {item_id} not found")
        audio_url = self._client.audio_url(
            jellyfin_track[ITEM_KEY_ID], container=SUPPORTED_CONTAINER_FORMATS
        )
        url = _normalize_jellyfin_media_url(audio_url)
        runtime_ticks = jellyfin_track.get(ITEM_KEY_RUNTIME_TICKS)
        return StreamDetails(
            item_id=jellyfin_track[ITEM_KEY_ID],
            provider=self.instance_id,
            media_type=media_type,
            audio_format=audio_format(jellyfin_track),
            stream_type=StreamType.HTTP,
            duration=(
                int(runtime_ticks / 10000000)  # 10000000 ticks per second
                if runtime_ticks is not None
                else None
            ),
            path=url,
            can_seek=True,
            allow_seek=True,
        )

    @use_cache(3600)  # Cache for 1 hour
    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Retrieve a dynamic list of tracks based on the provided item."""
        resp = await self._client.get_similar_tracks(
            prov_track_id, limit=limit, fields=TRACK_FIELDS
        )
        return [
            parse_track(self.logger, self.instance_id, self._client, track)
            for track in resp["Items"]
        ]

    async def _get_music_libraries(self) -> list[JellyMediaLibrary]:
        """Return all supported libraries a user has access to."""
        response = await self._client.get_media_folders()
        libraries = response["Items"]
        result = []
        for library in libraries:
            if library.get(ITEM_KEY_COLLECTION_TYPE) == COLLECTION_TYPE_MUSIC:
                result.append(library)
        return result

    async def _get_playlists(self) -> list[JellyMediaLibrary]:
        """Return all supported libraries a user has access to."""
        response = await self._client.get_media_folders()
        libraries = response["Items"]
        result = []
        for library in libraries:
            if library.get(ITEM_KEY_COLLECTION_TYPE) == COLLECTION_TYPE_PLAYLISTS:
                result.append(library)
        return result

    @property
    def _progress_poll_task_id(self) -> str:
        return f"jellyfin_audiobook_progress_{self.instance_id}"

    def _schedule_audiobook_progress_poll(self) -> None:
        self.mass.call_later(
            AUDIOBOOK_PROGRESS_POLL_SECONDS,
            self._poll_audiobook_progress,
            task_id=self._progress_poll_task_id,
        )

    async def _poll_audiobook_progress(self) -> None:
        """Pick up audiobook progress once, and schedule the next time."""
        try:
            await self._sync_audiobook_progress()
        except Exception as err:
            # anything unlisted would end the polling for good, with resume still working,
            # which leaves "continue listening" quietly out of date until the next restart
            self.logger.warning("Could not check audiobook progress: %s", err)
        if not self.unloading:
            self._schedule_audiobook_progress_poll()

    async def _sync_audiobook_progress(self) -> None:
        """
        Reflect audiobook positions changed in other Jellyfin clients in the playlog.

        This is what makes a book listened to elsewhere the most recent one in Music Assistant.
        Only changes are pushed, because each push counts as playing the book now.
        """
        changed = False
        for library in await self._get_book_libraries():
            async for book in self._iter_raw_audiobooks(library[ITEM_KEY_ID]):
                changed = await self._push_audiobook_progress(book) or changed
        if changed:
            await self._store_seen_progress()

    async def _push_audiobook_progress(self, book: dict[str, Any]) -> bool:
        """
        Push the progress of one book to the playlog if it changed.

        :param book: The raw Jellyfin item.
        :return: Whether the seen progress changed.
        """
        item_id = book[ITEM_KEY_ID]
        user_data = book.get(ITEM_KEY_USER_DATA) or {}
        ticks = int(user_data.get(USER_DATA_KEY_POSITION_TICKS) or 0)
        played = bool(user_data.get(USER_DATA_KEY_PLAYED, False))
        seen = self._seen_progress.get(item_id)
        if seen == [ticks, played]:
            return False
        untouched = not ticks and not played
        if seen is None and untouched:
            self._seen_progress[item_id] = [ticks, played]
            return True
        library_item = await self.mass.music.get_library_item_by_prov_id(
            media_type=MediaType.AUDIOBOOK,
            item_id=item_id,
            provider_instance_id_or_domain=self.instance_id,
        )
        if library_item is None:
            # not synced yet (or skipped): stays unseen, so it is tried again next time
            return False
        if untouched:
            await self.mass.music.mark_item_unplayed(
                library_item, provider_instance_id=self.instance_id
            )
        else:
            await self.mass.music.mark_item_played(
                library_item,
                fully_played=played,
                seconds_played=(
                    cast("Audiobook", library_item).duration
                    if played
                    else ticks // TICKS_PER_SECOND
                ),
                user_initiated=False,
                provider_instance_id=self.instance_id,
            )
        self._seen_progress[item_id] = [ticks, played]
        return True

    async def _store_seen_progress(self) -> None:
        await self.mass.cache.set(
            CACHE_KEY_SEEN_PROGRESS,
            self._seen_progress,
            expiration=CACHE_EXPIRATION_SEEN_PROGRESS,
            provider=self.instance_id,
            persistent=True,
        )

    async def _remember_known_ticks(self, item_id: str, ticks: int) -> None:
        self._known_ticks[item_id] = ticks
        await self.mass.cache.set(
            CACHE_KEY_KNOWN_TICKS,
            self._known_ticks,
            expiration=CACHE_EXPIRATION_SEEN_PROGRESS,
            provider=self.instance_id,
            persistent=True,
        )

    async def _moved_elsewhere(self, item_id: str) -> bool:
        """
        Return whether another client moved the position since we last read or wrote it.

        :param item_id: Id of the audiobook.
        """
        known = self._known_ticks.get(item_id)
        if known is None:
            return False
        book = await self._get_raw_audiobook(item_id)
        user_data = book.get(ITEM_KEY_USER_DATA) or {}
        return int(user_data.get(USER_DATA_KEY_POSITION_TICKS) or 0) != known

    async def _post_user_data(self, item_id: str, data: dict[str, Any]) -> None:
        """
        Update the user data of an item for the signed-in user.

        :param item_id: Id of the item.
        :param data: The user data fields to set; other fields are left alone.
        """
        config = self._client._session_config
        async with config.session.post(
            f"{self._client.base_url}/UserItems/{item_id}/UserData",
            params={"userId": self._client._user_id},
            json=data,
            headers={
                "User-Agent": config.user_agent,
                "Authorization": config.authentication_header(self._client._access_token),
            },
            ssl=config.verify_ssl,
            raise_for_status=True,
        ):
            pass

    async def _iter_raw_audiobooks(self, library_id: str) -> AsyncGenerator[dict[str, Any]]:
        """
        Yield the raw AudioBook items of a library.

        Raw JSON on purpose: aiojellyfin's typed decoders lose People and Chapters.

        :param library_id: Id of the Jellyfin library.
        """
        offset = 0
        while True:
            page = await self._client._session.get_json(
                "/Items",
                params={
                    "parentId": library_id,
                    "includeItemTypes": ITEM_TYPE_AUDIOBOOK,
                    "recursive": "true",
                    "enableUserData": "true",
                    "fields": ",".join(AUDIOBOOK_FIELDS),
                    "limit": str(AUDIOBOOK_PAGE_SIZE),
                    "startIndex": str(offset),
                },
            )
            for book in page["Items"]:
                yield book
            offset += AUDIOBOOK_PAGE_SIZE
            if offset >= page["TotalRecordCount"]:
                return

    async def _get_raw_audiobook(self, item_id: str) -> dict[str, Any]:
        """
        Return the raw item of an audiobook.

        :param item_id: Id of the audiobook.
        :raises MediaNotFoundError: If there is no such item, or it is not an audiobook.
        """
        try:
            item = await self._client._session.get_json(
                f"/Users/{self._client._user_id}/Items/{item_id}",
                params={"fields": ",".join(AUDIOBOOK_FIELDS), "enableUserData": "true"},
            )
        except (NotFound, SessionNotFound) as err:
            # aiojellyfin raises its session's NotFound here, which is a different class
            raise MediaNotFoundError(f"Item {item_id} not found") from err
        if item.get(ITEM_KEY_TYPE) != ITEM_TYPE_AUDIOBOOK:
            raise MediaNotFoundError(f"Item {item_id} is not an audiobook")
        return item

    async def _get_book_libraries(self) -> list[JellyMediaLibrary]:
        """Return all books libraries the user has access to."""
        response = await self._client.get_media_folders()
        return [
            library
            for library in response["Items"]
            if library.get(ITEM_KEY_COLLECTION_TYPE) == COLLECTION_TYPE_BOOKS
        ]


def _normalize_jellyfin_media_url(url: str) -> str:
    """Return a media URL using Jellyfin's supported auth parameter."""
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key.lower() == "api_key" for key, _ in query):
        return url

    query = [("apiKey" if key.lower() == "api_key" else key, value) for key, value in query]
    return urlunsplit(parsed._replace(query=urlencode(query)))
