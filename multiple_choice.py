"""Automatic numbering for multiple choice questions with sampling of style."""
from collections.abc import Iterable, Sequence
import random


class MultipleChoiceStyle:
    """Automatic numbering."""

    def __init__(self, markers: list[str]) -> None:
        self._markers = markers

    def markers_for_options(self, options: Sequence[str]) -> list[str]:
        """Returns a list of markers for the given options."""
        if len(options) > len(self._markers):
            raise ValueError(
                f'Not enough markers for {len(options)} options: {options}')
        return self._markers[:len(options)]


_STYLES = [
    MultipleChoiceStyle(['(1)', '(2)', '(3)', '(4)', '(5)']),
    MultipleChoiceStyle(['1)', '2)', '3)', '4)', '5)']),
    MultipleChoiceStyle(['1.', '2.', '3.', '4.', '5.']),
    MultipleChoiceStyle(['1', '2', '3', '4', '5']),

    MultipleChoiceStyle(['(A)', '(B)', '(C)', '(D)', '(E)']),
    MultipleChoiceStyle(['A)', 'B)', 'C)', 'D)', 'E)']),
    MultipleChoiceStyle(['A.', 'B.', 'C.', 'D.', 'E.']),
    MultipleChoiceStyle(['A', 'B', 'C', 'D', 'E']),

    MultipleChoiceStyle(['(a)', '(b)', '(c)', '(d)', '(e)']),
    MultipleChoiceStyle(['a)', 'b)', 'c)', 'd)', 'e)']),

    MultipleChoiceStyle(['(i)', '(ii)', '(iii)', '(iv)', '(v)']),
    MultipleChoiceStyle(['i)', 'ii)', 'iii)', 'iv)', 'v)']),

    MultipleChoiceStyle(['(I)', '(II)', '(III)', '(IV)', '(V)']),
    MultipleChoiceStyle(['I)', 'II)', 'III)', 'IV)', 'V)']),
]


def sample_markers_for_options(options: Sequence[str]) -> list[str]:
    """Return the markers from a randomly sampled style."""
    style = random.choice(_STYLES)
    return style.markers_for_options(options)
