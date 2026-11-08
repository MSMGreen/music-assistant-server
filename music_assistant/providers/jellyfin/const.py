"""Constants for the Jellyfin integration."""

from typing import Final

from aiojellyfin import ImageType as JellyImageType
from aiojellyfin import ItemFields
from music_assistant_models.enums import ImageType

DOMAIN: Final = "jellyfin"

COLLECTION_TYPE_MUSIC: Final = "music"
COLLECTION_TYPE_PLAYLISTS: Final = "playlists"
COLLECTION_TYPE_BOOKS: Final = "books"

ITEM_KEY_COLLECTION_TYPE: Final = "CollectionType"
ITEM_KEY_ID: Final = "Id"
ITEM_KEY_IMAGE_TAGS: Final = "ImageTags"
ITEM_KEY_INDEX_NUMBER: Final = "IndexNumber"
ITEM_KEY_MEDIA_SOURCES: Final = "MediaSources"
ITEM_KEY_MEDIA_TYPE: Final = "MediaType"
ITEM_KEY_MEDIA_STREAMS: Final = "MediaStreams"
ITEM_KEY_MEDIA_CHANNELS: Final = "Channels"
ITEM_KEY_MEDIA_CODEC: Final = "Codec"
ITEM_KEY_MEDIA_STREAM_TYPE: Final = "Type"
ITEM_KEY_CONTAINER: Final = "Container"
ITEM_KEY_NAME: Final = "Name"
ITEM_KEY_PROVIDER_IDS: Final = "ProviderIds"
ITEM_KEY_PRODUCTION_YEAR: Final = "ProductionYear"
ITEM_KEY_OVERVIEW: Final = "Overview"
ITEM_KEY_MUSICBRAINZ_RELEASE_GROUP: Final = "MusicBrainzReleaseGroup"
ITEM_KEY_MUSICBRAINZ_ARTIST: Final = "MusicBrainzArtist"
ITEM_KEY_MUSICBRAINZ_ALBUM: Final = "MusicBrainzAlbum"
ITEM_KEY_MUSICBRAINZ_TRACK: Final = "MusicBrainzTrack"
ITEM_KEY_SORT_NAME: Final = "SortName"
ITEM_KEY_CHAPTERS: Final = "Chapters"
ITEM_KEY_PARENT_ID: Final = "ParentId"
ITEM_KEY_PEOPLE: Final = "People"
ITEM_KEY_ALBUM_ARTIST: Final = "AlbumArtist"
ITEM_KEY_ALBUM_ARTISTS: Final = "AlbumArtists"
ITEM_KEY_ALBUM: Final = "Album"
ITEM_KEY_ALBUM_ID: Final = "AlbumId"
ITEM_KEY_ARTIST_ITEMS: Final = "ArtistItems"
ITEM_KEY_CAN_DOWNLOAD: Final = "CanDownload"
ITEM_KEY_PARENT_INDEX_NUM: Final = "ParentIndexNumber"
ITEM_KEY_RUNTIME_TICKS: Final = "RunTimeTicks"
ITEM_KEY_TYPE: Final = "Type"
ITEM_KEY_USER_DATA: Final = "UserData"

USER_DATA_KEY_IS_FAVORITE: Final = "IsFavorite"
USER_DATA_KEY_LAST_PLAYED: Final = "LastPlayedDate"
USER_DATA_KEY_PLAYED: Final = "Played"
USER_DATA_KEY_POSITION_TICKS: Final = "PlaybackPositionTicks"

CHAPTER_KEY_NAME: Final = "Name"
CHAPTER_KEY_START_TICKS: Final = "StartPositionTicks"

PERSON_KEY_NAME: Final = "Name"
PERSON_KEY_TYPE: Final = "Type"
PERSON_TYPE_AUTHOR: Final = "Author"
PERSON_TYPE_NARRATOR: Final = "Narrator"

TICKS_PER_SECOND: Final = 10_000_000
TICKS_PER_MS: Final = 10_000

MEDIA_TYPE_AUDIO: Final = "Audio"
ITEM_TYPE_AUDIOBOOK: Final = "AudioBook"
AUDIOBOOK_PAGE_SIZE: Final = 100
# how often positions changed by other Jellyfin clients are picked up; library sync is hours
AUDIOBOOK_PROGRESS_POLL_SECONDS: Final = 300
CACHE_KEY_SEEN_PROGRESS: Final = "audiobook_progress_seen"
CACHE_KEY_KNOWN_TICKS: Final = "audiobook_known_ticks"
CACHE_KEY_PLAYBACK_OWNER: Final = "audiobook_playback_owner"
CACHE_EXPIRATION_SEEN_PROGRESS: Final = 365 * 24 * 3600

SUPPORTED_CONTAINER_FORMATS: Final = "ogg,flac,mp3,aac,mpeg,alac,wav,aiff,wma,m4a,m4b,dsf,opus,wv"

ARTIST_FIELDS: Final = [
    ItemFields.Overview,
    ItemFields.ProviderIds,
    ItemFields.SortName,
]
ALBUM_FIELDS: Final = [
    ItemFields.Overview,
    ItemFields.ProviderIds,
    ItemFields.SortName,
]
TRACK_FIELDS: Final = [
    ItemFields.ProviderIds,
    ItemFields.CanDownload,
    ItemFields.SortName,
    ItemFields.MediaSources,
    ItemFields.MediaStreams,
]

AUDIOBOOK_FIELDS: Final = [
    ItemFields.Chapters,
    ItemFields.MediaSources,
    ItemFields.MediaStreams,
    ItemFields.People,
    ItemFields.Overview,
    ItemFields.SortName,
    ItemFields.ProviderIds,
]

USER_APP_NAME: Final = "Music Assistant"

MEDIA_IMAGE_TYPES: Final = {
    JellyImageType.Primary: ImageType.THUMB,
    JellyImageType.Logo: ImageType.LOGO,
}
