"""
backend.favourites package
"""

from .schemas import (
    AddFavouriteRequest,
    RemoveFavouriteRequest,
    ShortlistCarItem,
    ShortlistResponse,
    ShortlistComparisonResponse,
)
from .service import FavouritesService

__all__ = [
    "AddFavouriteRequest",
    "RemoveFavouriteRequest",
    "ShortlistCarItem",
    "ShortlistResponse",
    "ShortlistComparisonResponse",
    "FavouritesService",
]

