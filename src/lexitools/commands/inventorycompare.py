"""
Run a phonological inventory comparison.
"""

import itertools
from collections import defaultdict
import csv

# Import MPI-SHH libraries
import pyclts
import pyglottolog
from pyclts.inventories import Inventory
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

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)


def jac(set_a, set_b):
    """
    Compute the Jaccard Index.
    """

    if len(set_a) == 0 or len(set_b) == 0:
        index = 0
    else:
        index = len(set(set_a).intersection(set(set_b))) / len(
            set(set_a).union(set(set_b))
        )

    return index


def non_clts_comparison(inv_a, inv_b, name_a, name_b, prefix=None):
    """
    Collect info for non-BIPA inventories.
    """

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
    """
    Collect info for BIPA inventories.
    """

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
    aspect_groups = {"all": None, "consonant": ["consonant"], "vowel": ["vowel"]}
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


def collect_inventory_values(dataset, inventory_list, parameter_map):
    """
    Collect inventories from a dataset.
    """

    # Collect raw/unicode/clts for all relevant inventories
    to_collect = []
    for catalog in inventory_list.keys():
        to_collect += list(
            itertools.chain.from_iterable(inventory_list[catalog].values())
        )

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


def collect_results(
    values_a, values_b, catalog_a, catalog_b, inventory_a, inventory_b, glottocode, bipa
):
    """
    Collect raw, unicode, and bipa comparison values for output.
    """

    output_row = {
        "glottocode": glottocode,
        "catalog_a": catalog_a,
        "catalog_b": catalog_b,
        "inventory_a": inventory_a,
        "inventory_b": inventory_b,
    }

    # Single-pass collection of all entries would be faster
    raw_a = sorted([entry["raw"] for entry in values_a])
    raw_b = sorted([entry["raw"] for entry in values_b])
    unicode_a = sorted({entry["unicode"] for entry in values_a})
    unicode_b = sorted({entry["unicode"] for entry in values_b})
    bipa_a = sorted({entry["bipa"] for entry in values_a if entry["bipa"]})
    bipa_b = sorted({entry["bipa"] for entry in values_b if entry["bipa"]})

    # Build output
    output_row.update(non_clts_comparison(raw_a, raw_b, catalog_a, catalog_b, "raw"))
    output_row.update(
        non_clts_comparison(unicode_a, unicode_b, catalog_a, catalog_b, "unicode")
    )

    clts_inv_a = Inventory.from_list(*bipa_a, clts=bipa)
    clts_inv_b = Inventory.from_list(*bipa_b, clts=bipa)
    output_row.update(
        clts_comparison(clts_inv_a, clts_inv_b, catalog_a, catalog_b, "bipa")
    )

    return output_row


def write_output(output_rows, catalog_a, catalog_b):
    """
    Write results for a pairwise catalog comparison.
    """

    first_fields = ["glottocode", "language"]

    filename = f"output/results_{catalog_a}-{catalog_b}.tsv"
    with open(filename, "w") as tsvfile:
        # Get fieldnames, making sure glottolog info comes firts
        fieldnames = list(output_rows[0].keys())
        fieldnames = first_fields + [
            field for field in fieldnames if field not in first_fields
        ]

        # Write results
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_soundtable(values, inventories, glottolog):
    """
    Write a single table with all sounds.
    """

    # Collect all sounds in a single table, so we can check when there is a missing BIPA;
    # This involves building an inverse inventory map
    inv_inventory = {}
    for catalog, inv_dict in inventories.items():
        for glottocode, inv_list in inv_dict.items():
            for inv_id in inv_list:
                inv_inventory[inv_id] = glottocode

    glottocodes = set()
    rows = []
    for inventory_id, sounds in values.items():
        glottocode = inv_inventory.get(inventory_id, None)
        glottocodes.add(glottocode)
        catalog = inventory_id.split("_")[0]
        for sound in sounds:
            rows.append(
                {
                    "catalog": catalog,
                    "inventory": inventory_id,
                    "glottocode": glottocode,
                    "raw": sound["raw"],
                    "unicode": sound["unicode"],
                    "bipa": sound["bipa"] if sound["bipa"] else "",
                }
            )

    # cache language names
    lang_names = {}
    for languoid in glottolog.languoids():
        if languoid.glottocode in glottocodes:
            lang_names[languoid.glottocode] = languoid.name

    # Add language names, sort, and output
    for row in rows:
        row["language"] = lang_names.get(row["glottocode"], "")
        t = (row["catalog"], row["language"], row["bipa"], row["unicode"])
        if None in t:
            print(row)
    rows = sorted(
        rows, key=lambda r: (r["catalog"], r["language"], r["bipa"], r["unicode"])
    )

    with open("output/sound-table.tsv", "w") as output:
        writer = csv.DictWriter(
            output,
            delimiter="\t",
            fieldnames=[
                "catalog",
                "language",
                "inventory",
                "glottocode",
                "raw",
                "unicode",
                "bipa",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    """
    Entry point for command-line call.
    """

    # Instantiate BIPA and Glottolog
    args.log.info("Instantiating CLTS and Glottolog...")
    bipa = pyclts.CLTS(args.clts.dir).bipa
    glottolog = pyglottolog.Glottolog(args.glottolog.dir)

    args.log.info("Loading dataset...")
    ds = get_dataset(args.dataset).cldf_reader()

    # Collect list of catalogs for the comparison
    # TODO: allow to set from command-line
    catalogs = sorted({row["Catalog"] for row in ds["ValueTable"]})

    # Collect mapping of language ids to glottocodes
    glottocode_map = {row["ID"]: row["Glottocode"] for row in ds["LanguageTable"]}

    # Collect parameters, which include unicode and bipa
    parameter_map = {
        row["ID"]: {"unicode": row["Name"], "bipa": row["BIPA"]}
        for row in ds["ParameterTable"]
    }

    # Collect all contribution IDs mapped to a given glottocode
    inventories = defaultdict(lambda: defaultdict(set))
    for row in ds["ValueTable"]:
        if row["Catalog"] in catalogs:
            glottocode = glottocode_map.get(row["Language_ID"])
            if glottocode:
                inventories[row["Catalog"]][glottocode].add(row["Contribution_ID"])

    values = collect_inventory_values(ds, inventories, parameter_map)

    # Write all sounds in a single table
    args.log.info("Writing sound table...")
    write_soundtable(values, inventories, glottolog)

    return

    # Get a list of all glottocodes in common between all combinations
    for catalog_a, catalog_b in itertools.combinations_with_replacement(catalogs, 2):
        glottocodes_a = list(inventories[catalog_a].keys())
        glottocodes_b = list(inventories[catalog_b].keys())

        common = sorted(
            [glottocode for glottocode in glottocodes_a if glottocode in glottocodes_b]
        )

        args.log.info(
            f"Processing `{catalog_a}` and `{catalog_b}` ({len(common)} common codes)..."
        )

        output_rows = []
        for glottocode in common:
            # If we are doing in-catalog comparison, we only compare glottocodes
            # that have multiple inventories
            if catalog_a == catalog_b:
                catalog_inventories = inventories[catalog_a][glottocode]
                if len(catalog_inventories) == 1:
                    comparanda = []
                else:
                    comparanda = list(itertools.combinations(catalog_inventories, 2))
            else:
                # Grab all inventories
                inventories_a = list(inventories[catalog_a][glottocode])
                inventories_b = list(inventories[catalog_b][glottocode])

                comparanda = list(itertools.product(inventories_a, inventories_b))

            for inventory_a, inventory_b in comparanda:
                # Extract values and build raw/unicode/bipa lists; we take sets of
                # unicode and clts, as we might have repeated items due to conversion
                values_a = values[inventory_a]
                values_b = values[inventory_b]

                # Collect row of output and add other info
                output_row = collect_results(
                    values_a,
                    values_b,
                    catalog_a,
                    catalog_b,
                    inventory_a,
                    inventory_b,
                    glottocode,
                    bipa,
                )
                output_row["language"] = glottolog.languoid(glottocode).name

                output_rows.append(output_row)

        # write results
        if output_rows:
            write_output(output_rows, catalog_a, catalog_b)
