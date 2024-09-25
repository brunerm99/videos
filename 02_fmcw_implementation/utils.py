"""This module contains some utility functions related
with Code() objects."""

from manim import Code
from manim.mobject.text.text_mobject import remove_invisible_chars


def find_in_code(code: Code, string: str, lines: int | list[int] | None = None):
    """
    Finds all occurrences of the string `string` in the code object `code`,
    optionally restricting the search to the lines specified in the `lines`
    parameter. In the lines parameter, the lines start numbered at 1
    to match the line numbers shown in the code object.

    The result is a list of tuples `(line_number, start, end)`
    where `line_number` is the line number (starting at 0), and
    `start` and `end` are the limits of a slice which contains the
    glyphs that match the string. Note that in the result the line
    numbers start at 0, not at 1 as in the `lines` parameter, to make
    easy to use the result with the `code` object.
    The result can be used to select the matching glyphs and apply
    a style or animation to them.

    Example:
    ```python
    result = find_in_code(code, "foo", lines=[1, 3])
    for line_number, start, end in result:
        code.code[line_number][start:end].set_color(RED)
    ```
    """
    if not str:
        return []
    # Remove invisible characters and split the code into lines
    # skipping the initial blank ones
    code.code = remove_invisible_chars(code.code)
    src_lines = code.code_string.split("\n")
    while not src_lines[0]:
        src_lines = src_lines[1:]

    # Build the list of lines to process
    lines_to_process: list[int]
    if lines is None:
        lines_to_process = range(1, len(src_lines) + 1)
    elif isinstance(lines, int):
        lines_to_process = [lines]
    else:
        lines_to_process = lines

    # Result list
    result = []

    for line_number, line in enumerate(src_lines):
        if line_number + 1 not in lines_to_process:
            continue
        words = line.split()
        start = 0
        for w in words:
            w_end = 0
            while True:
                w_start = w.find(string, w_end)
                if w_start == -1:
                    break
                w_end = w_start + len(string)
                result.append((line_number, start + w_start, start + w_end))
            start += len(w)
    return result
