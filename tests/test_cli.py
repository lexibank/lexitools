import logging
import pathlib

import pytest

# lexitools commands are called as subcommands of cldfbench:
from cldfbench.__main__ import main as cldfbenchmain


@pytest.fixture
def main():
    """
    Fixture to abstract the way lexitools commands are actually called.

    :return: A `main` function which accepts arguments as positional args.
    """
    def func(cmd, *args):
        cldfbenchmain(
            ['lexitools.' + cmd] + [str(a) for a in args], log=logging.getLogger(__name__))
    return func


def test_example(main, capsys, tmpdir, caplog, test_data):
    # Write some test data to a temporary file:
    p = pathlib.Path(str(tmpdir)).joinpath('test.txt')
    p.write_text('k', encoding='utf8')

    # Call `lexitools.commands.example`:
    main('example', test_data / 'dataset' / 'lexibank_dataset.py', p)

    # and inspect what the command has written to stdout:
    out, _ = capsys.readouterr()
    assert out.strip().split() == [
        'dataset',  # That's the "print(ds.id)"
        '9',
    ]

    assert caplog.records[-1].levelname == "WARNING"
