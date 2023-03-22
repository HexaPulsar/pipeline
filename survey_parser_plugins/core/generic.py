import abc
import functools
from dataclasses import dataclass, field
from typing import Sequence, Set

from .mapper import Mapper


@dataclass
class GenericAlert:
    """Alert of astronomical surveys."""

    oid: str  # name of object (from survey)
    tid: str  # telescope identifier
    pid: int  # processing identifier for image
    candid: str  # candidate identifier (from survey)
    mjd: float  # modified Julian date
    fid: int  # filter identifier
    ra: float  # right ascension
    dec: float  # declination
    mag: float  # difference magnitude
    e_mag: float  # difference magnitude uncertainty
    isdiffpos: int  # sign of the flux difference
    e_ra: float = None  # right ascension uncertainty
    e_dec: float = None  # declination uncertainty
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return self.__getattribute__(item)


class SurveyParser(abc.ABC):
    """Base class for survey parsing. Subclasses are intended to be static.

    The field `_source` is an identifier for the survey.

    The field `_mapping` should have a list of `Mapper` objects and have one entry per field in `GenericAlert`,
    except for `stamps` and `extra_fields`, which are generated through specialized methods.

    The field `_ignore_in_extra_fields` should include fields from the message that are not added to `extra_fields`.
    In addition to the fields described above, the parser automatically ignores all the `origin` fields from the
    `Mapper` objects described in `_mapping`.
    """
    _source: str
    _mapping: Sequence[Mapper]
    _ignore_in_extra_fields: Sequence[str] = []

    @classmethod
    @functools.lru_cache(1)
    def _exclude_from_extra_fields(cls) -> Set[str]:
        """Returns a set of fields that should not be present in `extra_fields` for `GenericAlert`"""
        ignore = {mapper.origin for mapper in cls._mapping if mapper.origin is not None}
        ignore.update(cls._ignore_in_extra_fields)
        return ignore

    @classmethod
    @abc.abstractmethod
    def _extract_stamps(cls, message: dict) -> dict:
        """Keys are `cutoutScience`, `cutoutTemplate` and `cutoutDifference`. Values are of type `byte` or `None`"""
        return {
            "cutoutScience": None,
            "cutoutTemplate": None,
            "cutoutDifference": None,
        }

    @classmethod
    def parse_message(cls, message: dict) -> GenericAlert:
        """Create a `GenericAlert` from the message"""
        generic = {mapper.field: mapper(message) for mapper in cls._mapping}

        stamps = cls._extract_stamps(message)
        extra_fields = {k: v for k, v in message.items() if k not in cls._exclude_from_extra_fields()}
        return GenericAlert(**generic, stamps=stamps, extra_fields=extra_fields)

    @classmethod
    @abc.abstractmethod
    def can_parse(cls, message: dict) -> bool:
        """Whether the message can be parsed"""

    @classmethod
    def get_source(cls) -> str:
        """Name of the parser source"""
        return cls._source
