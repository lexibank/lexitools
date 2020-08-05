"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import PathType
from cldfbench.cli_util import add_catalog_spec, get_dataset
from tabulate import tabulate
from pylexibank import progressbar
from collections import defaultdict

from linse.annotate import prosody, clts
from linse.transform import syllables, morphemes

from pylexibank.cli_util import add_dataset_spec


def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)

    # Add another argument:
    #parser.add_argument(
    #        '--medials',
    #        action='store',
    #        default=None,
    #        help='define your medials'
    #        )
    parser.add_argument(
            '--doculect',
            action='store',
            default=None,
            help='select one doculect'
            )
    parser.add_argument(
            '--display',
            action='store',
            default=None,
            help='select a display')


def run(args):

    ds = get_dataset(args)
    P = defaultdict(lambda : defaultdict(list))
    if ds.cldf_dir.joinpath("forms.csv").exists():
        for row in progressbar(
                ds.cldf_reader()['FormTable'], 
                desc='iterate over wordlist'):
            if row['Language_ID'] == args.doculect or not args.doculect:
                if row['Language_ID'] not in P:
                    P[row['Language_ID']] = defaultdict(lambda : defaultdict(list))
                tmp = P[row['Language_ID']] # consider not doing this
                for morpheme in morphemes(row['Segments']):
                    for syl in syllables(morpheme):
                        cv = prosody(syl, format='CcV')
                        template = ''.join(cv)
                        for i, (s, c) in enumerate(zip(syl, cv)):
                            this_template = template[:i]+'**'+template[i]+'**'+template[i+1:]
                            tmp[s][this_template] += [row['ID']]
    table = []
    if args.display == 'long':
        header = [
            'Language', 'Sound', 'Template', 'Frequency']
        for language, data in P.items():
            for sound, templates in data.items():
                for template, frequency in templates.items():
                    table += [[language, sound, template, len(frequency)]]
        print(tabulate(table, headers=header, tablefmt='pipe'))
    else:
        header = [
            'Language', 'Sound', 'Class', 'Frequency', 'Templates']
        for language, data in P.items():
            for sound, templates in data.items():
                table += [[
                    language, 
                    sound, 
                    clts(sound.split('/')[1] if '/' in sound else sound)[0].split()[-1],
                    sum([len(x) for x in templates.values()]), 
                    ', '.join(['{0}:{1}'.format(x, len(y)) for x, y in templates.items()])]]
        print(tabulate(
            sorted(table, key=lambda x: (x[2], x[3], x[1]), reverse=True), 
            headers=header, tablefmt='pipe'))
