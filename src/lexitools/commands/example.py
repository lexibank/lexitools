"""
An example command, serving as instruction on how to write commands.
"""
from clldutils.clilib import PathType
from cldfbench.cli_util import add_catalog_spec, get_dataset

# `pylexibank.cli_util.add_dataset_spec` looks for the correct EntryPoint by default:
from pylexibank.cli_util import add_dataset_spec


def register(parser):
    """
    Register command options and arguments.

    :param parser: and `argparse.ArgumentParser`instance.
    """
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, 'clts')

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)

    # Add a flag (i.e. a boolean option):
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='do stuff in a strict way',
    )

    # Add another argument:
    parser.add_argument(
        'input_file',
        type=PathType(type='file'),
        help='some input from a file',
    )


def run(args):
    # Access the dataset:
    ds = get_dataset(args)
    print(ds.id)
    # and its CLDF Dataset:
    print(len(list(ds.cldf_reader()['LanguageTable'])))

    # Thanks to `PathType` `args.input_file` is a `pathlib.Path`:
    for c in args.input_file.read_text(encoding='utf8'):
        if args.strict:  # evaluates our flag
            # The CLTS catalog API is available as `args.clts.api`:
            print(args.clts.api.bipa[c].name)  # pragma: no cover
        else:
            args.log.warning('not very strict')
