"""
Run a phonological inventory comparison.
"""

from collections import defaultdict

# Import MPI-SHH libraries
import pyclts
import pyglottolog
from pyclts.inventories import Inventory
from clldutils.clilib import PathType
from cldfbench import get_dataset
from cldfbench.cli_util import add_catalog_spec
from pylexibank.cli_util import add_dataset_spec


def register(parser):
    """
    Register command options and arguments.

    :param parser: and `argparse.ArgumentParser`instance.
    """

    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")
    add_catalog_spec(parser, "glottolog")

    # Datasets for comparison
    parser.add_argument("ds1", type=str, help="Entry point for Dataset 1")
    parser.add_argument("ds2", type=str, help="Entry point for Dataset 2")

    # Results
    parser.add_argument("output", type=str, help="Output file")


# Get a dictionary of glottocodes/IDs for comparison
# NOTE: will fail if there is more than one ID for the same glottocode
def get_glottocodes(dataset):
    return {entry["Glottocode"]: entry["ID"] for entry in dataset["LanguageTable"]}


def get_inventories(dataset, bipa):
    inv = defaultdict(set)
    for row in list(dataset["ValueTable"]):
        sound = bipa[row["Value"]]
        if not isinstance(sound, pyclts.models.UnknownSound):
            inv[row["Language_ID"]].add(str(sound))

    return inv


def run(args):
    """
    Entry point for command-line call.
    """

    # Instantiate BIPA and Glottolog
    bipa = pyclts.CLTS(args.clts.dir).bipa
    glottolog = pyglottolog.Glottolog(args.glottolog.dir)

    # Get dataset readers
    args.log.info("Loading CLDF datasets...")
    ds1 = get_dataset(args.ds1).cldf_reader()
    ds2 = get_dataset(args.ds2).cldf_reader()

    # Get glottocodes/IDs for comparison
    lang1 = get_glottocodes(ds1)
    args.log.info("Dataset #1 has %i languages.", len(lang1))
    lang2 = get_glottocodes(ds2)
    args.log.info("Dataset #2 has %i languages.", len(lang2))

    # Get overlapping glottocodes
    overlap = [glottocode for glottocode in lang1 if glottocode in lang2]
    args.log.info("There are %i overlapping glottocodes.", len(overlap))

    # Get inventoris by ids
    invs1 = get_inventories(ds1, bipa)
    invs2 = get_inventories(ds2, bipa)

    # Compare all overlapping glottocodes
    with open(args.output, "w") as handler:
        header = [
            "Glottocode",
            "Language",
            "Strict_Similarity",
            "Approx_Similarity1",
            "Approx_Similarity2",
        ]
        handler.write("\t".join(header))
        handler.write("\n")

        for glottocode in sorted(overlap):
            args.log.info("Comparing inventories for glottocode `%s`...", glottocode)
            lang1_id = lang1[glottocode]
            lang2_id = lang2[glottocode]
            inv1 = Inventory.from_list(*list(invs1[lang1_id]), clts=bipa)
            inv2 = Inventory.from_list(*list(invs2[lang2_id]), clts=bipa)

            buf = [
                glottocode,
                glottolog.languoid(glottocode).name,
                "%.4f" % inv1.similar(inv2, metric="strict"),
                "%.4f" % inv1.similar(inv2, metric="approximate"),
                "%.4f" % inv2.similar(inv1, metric="approximate"),
            ]

            handler.write("\t".join(buf))
            handler.write("\n")
