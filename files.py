from __future__ import annotations

import numbers
import os
import pathlib
import types
from typing import Callable, Final, Protocol, TextIO

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

_MAX_FILE_SIZE: Final[float] = 1e9  # 1 GB


class SupportsWrite(Protocol):
    """For type-hinting anything  that can be written to."""

    def write(self, s: str) -> None:
        ...


class SequentialFileWriter:
    """Writes to a sequence of files."""

    def __init__(self,
                 get_file_name: Callable[[int], pathlib.Path],
                 max_size: numbers.Real = _MAX_FILE_SIZE,
                 check_max_file_size_programmatically: bool = False):
        """Writes to a sequence of files.

        Args:
            max_size: The maximum size of each file.
            get_file_name: A function that takes a file index and returns the
                corresponding file name.
        """
        if max_size <= 0:
            raise ValueError('max_size must be positive.')

        self._max_size: Final[numbers.Real] = max_size
        self._check_max_file_size_programmatically: bool = check_max_file_size_programmatically
        self._current_file_index = 0
        self._get_file_name: Final[Callable[[int], pathlib.Path]] = get_file_name
        self._current_file = self._open_new_file()
        # It's faster to track the current size ourselves than to call
        # os.path.getsize() every time.
        # On the other hand, since we use compressed files, the size of the final files may vary considerably.
        # It could lead to a huge number of small files.
        # Set `check_max_file_size_programmatically` to True,
        # to check the size of the file keeping a file size variable in memory.
        self._current_file_size: int = 0

    def _open_new_file(self) -> TextIO:
        self._filename = self._get_file_name(self._current_file_index)
        return xz.open(self._filename, 'wt')

    def _close_current_file(self) -> None:
        if self._current_file:
            self._current_file.close()
            self._current_file = None
            self._current_file_size = 0

    def _switch_to_new_file(self) -> None:
        self._close_current_file()
        self._current_file_index += 1
        self._current_file = self._open_new_file()

    def write(self, data: str) -> None:
        if self._check_max_file_size_programmatically:
            if self._current_file_size + len(data) > self._max_size:
                self._switch_to_new_file()
        else:
            if os.path.getsize(self._filename) > self._max_size:
                self._switch_to_new_file()

        self._current_file.write(data)
        self._current_file_size += len(data)

    def close(self) -> None:
        self._close_current_file()

    def __enter__(self) -> SequentialFileWriter:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: Exception | None,
            traceback: types.TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Implemented in case the caller forgets to call `close` or the
        # `with` statement is ended early.
        self.close()
