"""
Run a phonological inventory comparison.
"""

from itertools import chain
from collections import defaultdict, Counter
import pathlib
import csv

# Import MPI-SHH libraries
import pyclts
import pyglottolog
from pyclts.inventories import Inventory
from clldutils.clilib import PathType
from cldfbench import get_dataset
from cldfbench.cli_util import add_catalog_spec
from pylexibank.cli_util import add_dataset_spec
import pycldf.dataset

from clldutils.clilib import Table, add_format


def register(parser):
    """
    Register command options and arguments.

    :param parser: and `argparse.ArgumentParser`instance.
    """

    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")
    add_catalog_spec(parser, "glottolog")

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)
    add_format(parser, default="pipe")

    # Datasets for comparison


#    parser.add_argument("ds1", type=str, help="Entry point for Dataset 1")
#    parser.add_argument("ds2", type=str, help="Entry point for Dataset 2")

# Results
#    parser.add_argument("output", type=str, help="Output file")


def jac(a, b):
    return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))


def non_clts_comparison(inv_a, inv_b, name_a, name_b, prefix=None):
    # Set prefix and separator, if any
    if prefix:
        prefix = f"{prefix}_"
    else:
        prefix = ""

    # Get common sounds
    common = [snd for snd in inv_a if snd in inv_b]

    # Build output, computing what is necessary
    column = {}
    column[f"{prefix}size_{name_a}"] = len(inv_a)
    column[f"{prefix}size_{name_b}"] = len(inv_b)
    column[f"{prefix}inv_{name_a}"] = " ".join(inv_a)
    column[f"{prefix}inv_{name_b}"] = " ".join(inv_b)
    column[f"{prefix}shared"] = " ".join(common)
    column[f"{prefix}size_shared"] = len(common)
    column[f"{prefix}strict"] = jac(inv_a, inv_b)
    column[f"{prefix}exclusive_{name_a}"] = " ".join(
        [snd for snd in inv_a if snd not in common]
    )
    column[f"{prefix}exclusive_{name_b}"] = " ".join(
        [snd for snd in inv_b if snd not in common]
    )

    return column

def clts_comparison(inv_a, inv_b, name_a, name_b, prefix=None):
    # Set prefix and separator, if any
    if prefix:
        prefix = f"{prefix}_"
    else:
        prefix = ""

    column = {}

    # get counts
    inv_a_cons = len(inv_a.sounds["consonant"])
    inv_a_vowl = len(inv_a.sounds["vowel"])
    inv_b_cons = len(inv_b.sounds["consonant"])
    inv_b_vowl = len(inv_b.sounds["vowel"])
    column[f"{prefix}size_{name_a}_all"] = inv_a_cons + inv_a_vowl
    column[f"{prefix}size_{name_b}_all"] = inv_b_cons + inv_b_vowl
    column[f"{prefix}size_{name_a}_cons"] = inv_a_cons
    column[f"{prefix}size_{name_b}_cons"] = inv_b_cons
    column[f"{prefix}size_{name_a}_vowl"] = inv_a_vowl
    column[f"{prefix}size_{name_b}_vowl"] = inv_b_vowl

    # compute similarities
    aspect_groups = {
        "all": None,
        "consonant": ["consonant"],
        "vowel": ["vowel"],
    }
    for aspect_label, aspects in aspect_groups.items():
        column[f"{prefix}strict-{aspect_label}"] = inv_a.similar(
            inv_b, metric="strict", aspects=aspects
        )
        column[f"{prefix}appr-{name_a}-{name_b}-{aspect_label}"] = inv_a.similar(
            inv_b, metric="approximate", aspects=aspects
        )
        column[f"{prefix}appr-{name_b}-{name_a}-{aspect_label}"] = inv_b.similar(
            inv_a, metric="approximate", aspects=aspects
        )

    # Collect consonants, vowels from both inventories, for overlap
    sounds_a = {
        aspect: sorted(list(inv_a.sounds[aspect])) for aspect in ["consonant", "vowel"]
    }
    sounds_b = {
        aspect: sorted(list(inv_b.sounds[aspect])) for aspect in ["consonant", "vowel"]
    }
    common_cons = [
        cons for cons in sounds_a["consonant"] if cons in sounds_b["consonant"]
    ]
    common_vowl = [vowl for vowl in sounds_a["vowel"] if vowl in sounds_b["vowel"]]

    column[f"{prefix}inv_{name_a}"] = " ".join(
        sounds_a["consonant"] + sounds_a["vowel"]
    )
    column[f"{prefix}inv_{name_b}"] = " ".join(
        sounds_b["consonant"] + sounds_b["vowel"]
    )
    column[f"{prefix}shared_cons"] = " ".join(common_cons)
    column[f"{prefix}shared_vowl"] = " ".join(common_vowl)
    column[f"{prefix}size_shared_cons"] = len(common_cons)
    column[f"{prefix}size_shared_vowl"] = len(common_vowl)
    column[f"{prefix}size_shared_all"] = len(common_cons) + len(common_vowl)
    column[f"{prefix}exclusive_{name_a}_cons"] = " ".join(
        [cons for cons in sounds_a["consonant"] if cons not in common_cons]
    )
    column[f"{prefix}exclusive_{name_b}_cons"] = " ".join(
        [cons for cons in sounds_b["consonant"] if cons not in common_cons]
    )
    column[f"{prefix}exclusive_{name_a}_vowl"] = " ".join(
        [vowl for vowl in sounds_a["vowel"] if vowl not in common_vowl]
    )
    column[f"{prefix}exclusive_{name_b}_vowl"] = " ".join(
        [vowl for vowl in sounds_b["vowel"] if vowl not in common_vowl]
    )

    return column


def collect_inventories(dataset, inventory_list, parameter_map):
    # Collect raw/unicode/clts for all relevant inventories
    to_collect = []
    for catalog in inventory_list.keys():
        to_collect += list(chain.from_iterable(inventory_list[catalog].values()))

    values = defaultdict(list)
    for row in dataset["ValueTable"]:
        if row["Contribution_ID"] in to_collect:
            values[row["Contribution_ID"]].append(
                {
                    "raw": row["Value"],
                    "unicode": parameter_map[row["Parameter_ID"]]["unicode"],
                    "bipa": parameter_map[row["Parameter_ID"]]["bipa"],
                }
            )

    return values

def run(args):
    """
    Entry point for command-line call.
    """

    # Instantiate BIPA and Glottolog
    args.log.info("Instantiating CLTS and Glottolog...")
    bipa = pyclts.CLTS(args.clts.dir).bipa
    glottolog = pyglottolog.Glottolog(args.glottolog.dir)

    # TODO: load properly, without `cldf_reader()`
    args.log.info("Loading dataset...")
    ds = get_dataset(args.dataset).cldf_reader()

    # Collect mapping of language ids to glottocodes
    glottocode_map = {row["ID"]: row["Glottocode"] for row in ds["LanguageTable"]}

    # Collect parameters, which include unicode and bipa
    parameter_map = {
        row["ID"]: {"unicode": row["Name"], "bipa": row["BIPA"]}
        for row in ds["ParameterTable"]
    }

    catalog_a = "depi"
    catalog_b = "phoibleea"

    # Collect all contribution IDs mapped to a given glottocode
    inventories = defaultdict(lambda: defaultdict(set))
    for row in ds["ValueTable"]:
        if row["Catalog"] in [catalog_a, catalog_b]:
            glottocode = glottocode_map.get(row["Language_ID"])
            if glottocode:
                inventories[row["Catalog"]][glottocode].add(row["Contribution_ID"])

    values = collect_inventories(ds, inventories, parameter_map)

    # Get glottocodes in common
    glottocodes_a = list(inventories[catalog_a].keys())
    glottocodes_b = list(inventories[catalog_b].keys())
    glottocodes_common = [
        glottocode for glottocode in glottocodes_a if glottocode in glottocodes_b
    ]

    # Get comparable inventories per glottocode
    output = []
    for glottocode in sorted(glottocodes_common):
        # TODO: check for more than one inventory
        inventory_a = list(inventories[catalog_a][glottocode])[0]
        inventory_b = list(inventories[catalog_b][glottocode])[0]

        # Extract values and build raw/unicode/bipa lists; we take sets of
        # unicode and clts, as we might have repeated items due to conversion
        values_a = values[inventory_a]
        values_b = values[inventory_b]

        raw_a = sorted([entry['raw'] for entry in values_a])
        raw_b = sorted([entry['raw'] for entry in values_b])
        unicode_a = sorted(set([entry['unicode'] for entry in values_a]))
        unicode_b = sorted(set([entry['unicode'] for entry in values_b]))
        bipa_a = sorted(set([entry['bipa'] for entry in values_a if entry['bipa']]))
        bipa_b = sorted(set([entry['bipa'] for entry in values_b if entry['bipa']]))

        # Build output
        row = {'glottocode':glottocode}
        row.update(non_clts_comparison(raw_a, raw_b, catalog_a, catalog_b, "raw"))
        row.update(non_clts_comparison(unicode_a, unicode_b, catalog_a, catalog_b, "unicode"))

        clts_inv_a = Inventory.from_list(*bipa_a, clts=bipa)
        clts_inv_b = Inventory.from_list(*bipa_b, clts=bipa)
        row.update(clts_comparison(clts_inv_a, clts_inv_b, catalog_a, catalog_b, "bipa"))

        output.append(row)

    # write results
    filename = f"results_{catalog_a}-{catalog_b}.tsv"
    args.log.info(f"Writing results to `{filename}`...")
    with open(filename, "w") as tsvfile:
        fieldnames = list(output[0].keys())
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
