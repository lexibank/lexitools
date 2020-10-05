"""
check structure if the tokens are matching the template.
"""
from clldutils.clilib import PathType
from cldfbench.cli_util import add_catalog_spec, get_dataset
from tabulate import tabulate
from pylexibank import progressbar
from linse.annotate import seallable
from linse.transform import morphemes
from collections import defaultdict

from pylexibank.cli_util import add_dataset_spec


MEDIALS = {
        "j", "w", "jw", "wj", "i̯", "u̯", "i̯u̯", "u̯i̯", "ɥ",
        "l", "lj", "lʲ", "r", "rj", "rʲ", "rʷ", "lʷ",
        }


def register(parser):
    """
    Register command options and arguments.

    :param parser: and `argparse.ArgumentParser`instance.
    """
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)

    # Add another argument:


    parser.add_argument(
            '--medials',
            action='store',
            default=None,
            help='define your medials'
            )
    parser.add_argument(
            '--doculect',
            action='store',
            default=None,
            help='select one doculect'
            )


def get_structure(sequence, medials=None):
    """
    produce a list of structure tokens
    """
    out = []
    for m in morphemes(sequence):
        morph = [x.split('/')[-1] if '/' in x else x for x in m]
        out += [
                seallable(
                    morph,
                    medials=medials
                    )]
    return out


def run(args):
    """
    main function.
    """
    ds = get_dataset(args)
    if args.medials:
        args.medials = set(args.medials.split(','))
    errors = {
            'length': defaultdict(list), 
            'syllable': defaultdict(list),
            'missing': defaultdict(list)}
    if ds.cldf_dir.joinpath("forms.csv").exists():
        for row in progressbar(ds.cldf_reader()["FormTable"], desc='iterate over wordlist'):
            if row['Language_ID'] == args.doculect or not args.doculect:
                strucs = get_structure(row['Segments'], medials=args.medials or MEDIALS)
                for i, (struc, segments) in enumerate(
                        zip(strucs, morphemes(row['Segments']))):
                    if len(struc) != len(segments):
                        errors['length'][' '.join(segments), ' '.join(struc)] += [(row['ID'], i, row['Language_ID'], row['Form'],
                        row['Segments'])]
                    elif '?' in struc:
                        errors['missing'][' '.join(segments), ' '.join(struc)] += [(row['ID'], i, row['Language_ID'], row['Form'],
                        row['Segments'])]
                    elif not 'n' in struc or not 't' in struc:
                        errors['syllable'][' '.join(segments), ' '.join(struc)] += [(row['ID'], i, row['Language_ID'], row['Form'],
                        row['Segments'])]
    
    for error, errorname in [
            ('length', 'Length Errors'), ('missing', 'Missing Values'),
            ('syllable', 'Syllable Errors')]:
        if errors[error]:
            print('# '+errorname+'\n')
            table = []
            for i, ((segments, structure), examples) in enumerate(errors[error].items()):
                table += [[i+1, segments, structure, len(examples)]]
            print(tabulate(
                table, 
                tablefmt='pipe', 
                headers=['Number', 'Segments', 'Structure', 'Examples']
                ))
            print('')



