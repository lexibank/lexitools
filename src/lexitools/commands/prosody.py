"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import Table, add_format
from cldfbench.cli_util import add_catalog_spec, get_dataset

from linse.transform import syllable_inventories

from pylexibank.cli_util import add_dataset_spec


def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)
    add_format(parser, default='pipe')

    parser.add_argument(
            '--language-id',
            action='store',
            default=None,
            help='select one doculect'
            )
    parser.add_argument(
            '--display',
            action='store',
            default=None,
            help='select a display')
    parser.add_argument(
            '--prosody-format',
            action='store',
            default='CcV',
            help='select a format for the prosodic strings')


def run(args):

    ds = get_dataset(args)
    forms = []
    for row in ds.cldf_reader()['FormTable']:
        if row['Language_ID'] == args.language_id or not args.language_id:
            forms.append(row)

    P = syllable_inventories(forms, format=args.prosody_format)
    bipa = args.clts.from_config().api.transcriptionsystem_dict['bipa']

    table = []
    if args.display == 'long':
        header = [
            'Language', 'Sound', 'Template', 'Frequency']
        for language, data in P.items():
            for sound, templates in data.items():
                for template, frequency in templates.items():
                    table += [[language, sound, template, len(frequency)]]
    else:
        header = [
            'Language', 'Sound', 'Class', 'Frequency', 'Templates']
        for language, data in P.items():
            for sound, templates in data.items():
                table += [[
                    language, 
                    sound, 
                    bipa[sound].type,
                    sum([len(x) for x in templates.values()]), 
                    ', '.join(['{0}:{1}'.format(x, len(y)) for x, y in templates.items()])]]

    with Table(args, *header, rows=table):
        pass
