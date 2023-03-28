import enum
from typing import Any
from typing import Any


class _AutoName(enum.Enum):
    """Enum that overrides `enum.auto` to make value from name instead of from
    next int.

    https://docs.python.org/3.10/library/enum.html#using-automatic-values
    """

    def _generate_next_value_(name: str, start: int, count: int,
                              last_values: list[Any]) -> Any:
        return name


@enum.unique
class TaskType(_AutoName):
    """The different task types available."""
    # TODO is this detailed enough or do we need to distinguish topic classification from judgment prediction or NER from argument mining?
    TEXT_CLASSIFICATION = enum.auto()
    QUESTION_ANSWERING = enum.auto()
    SUMMARIZATION = enum.auto()
    NAMED_ENTITY_RECOGNITION = enum.auto()
    NATURAL_LANGUAGE_INFERENCE = enum.auto()
    MULTIPLE_CHOICE = enum.auto()
    ARGUMENTATION = enum.auto()
    QUESTION_GENERATION = enum.auto()
    CODE = enum.auto()
    UNKNOWN = enum.auto()


@enum.unique
class Jurisdiction(_AutoName):
    """The jurisdiction where cases are from."""
    # EU
    AUSTRIA = enum.auto()
    BELGIUM = enum.auto()
    BULGARIA = enum.auto()
    CROATIA = enum.auto()
    CZECHIA = enum.auto()
    DENMARK = enum.auto()
    ESTONIA = enum.auto()
    FINLAND = enum.auto()
    FRANCE = enum.auto()
    GERMANY = enum.auto()
    GREECE = enum.auto()
    HUNGARY = enum.auto()
    IRELAND = enum.auto()
    ITALY = enum.auto()
    LATVIA = enum.auto()
    LITHUANIA = enum.auto()
    LUXEMBOURG = enum.auto()
    MALTA = enum.auto()
    NETHERLANDS = enum.auto()
    POLAND = enum.auto()
    PORTUGAL = enum.auto()
    ROMANIA = enum.auto()
    SLOVAKIA = enum.auto()
    SLOVENIA = enum.auto()
    SPAIN = enum.auto()
    SWEDEN = enum.auto()
    # Europa
    EU = enum.auto()
    SWITZERLAND = enum.auto()
    UK = enum.auto()
    # Asia
    CHINA = enum.auto()
    INDIA = enum.auto()
    JAPAN = enum.auto()
    SOUTH_KOREA = enum.auto()
    THAILAND = enum.auto()
    # North America
    US = enum.auto()
    CANADA = enum.auto()
    # South America
    BRAZIL = enum.auto()
    # Other
    INTERNATIONAL = enum.auto()  # international law
    UNKNOWN = enum.auto()  # we don't know the jurisdiction
    N_A = enum.auto()  # Not a legal task
