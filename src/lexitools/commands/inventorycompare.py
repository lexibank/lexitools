"""
Run a phonological inventory comparison.
"""

from collections import defaultdict
import pathlib

# Import MPI-SHH libraries
import pyclts
import pyglottolog
from pyclts.inventories import Inventory
from clldutils.clilib import PathType
from cldfbench import get_dataset
from cldfbench.cli_util import add_catalog_spec
from pylexibank.cli_util import add_dataset_spec
import pycldf.dataset


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

    # Get dataset readers; as `get_dataset()` does not accept metadata files,
    # we check the arguments and load with `pycldf` if that is the case
    # TODO: deal with lexibank datasets, computing the inventories
    args.log.info("Loading CLDF datasets...")
    ds1_path = pathlib.Path(args.ds1).resolve()
    ds2_path = pathlib.Path(args.ds2).resolve()
    if ds1_path.name.endswith("-metadata.json"):
        ds1 = pycldf.dataset.StructureDataset.from_metadata(ds1_path)
    else:
        ds1 = get_dataset(args.ds1).cldf_reader()

    if ds2_path.name.endswith("-metadata.json"):
        ds2 = pycldf.dataset.StructureDataset.from_metadata(ds2_path)
    else:
        ds2 = get_dataset(args.ds1).cldf_reader()

    ds1_name = ds1.metadata_dict["rdf:ID"].upper()
    ds2_name = ds2.metadata_dict["rdf:ID"].upper()

    # Get glottocodes/IDs for comparison
    lang1 = get_glottocodes(ds1)
    args.log.info(f"Dataset #1 has {len(lang1)} languages.")
    lang2 = get_glottocodes(ds2)
    args.log.info(f"Dataset #2 has {len(lang2)} languages.")

    # Get overlapping glottocodes
    overlap = [glottocode for glottocode in lang1 if glottocode in lang2]
    args.log.info(f"There are {len(overlap)} overlapping glottocodes.")

    # Get inventoris by ids
    invs1 = get_inventories(ds1, bipa)
    invs2 = get_inventories(ds2, bipa)

    # Compare all overlapping glottocodes
    with open(args.output, "w") as handler:
        header = [
            "Glottocode",
            "Language",
            f"Size_{ds1_name}_ALL",
            f"Size_{ds2_name}_ALL",
            f"Size_{ds1_name}_C",
            f"Size_{ds2_name}_C",
            f"Size_{ds1_name}_V",
            f"Size_{ds2_name}_V",
            "Strict_Similarity_ALL",
            f"Approx_Similarity_{ds1_name}_ALL",
            f"Approx_Similarity_{ds2_name}_ALL",
            "Strict_Similarity_C",
            f"Approx_Similarity_{ds1_name}_C",
            f"Approx_Similarity_{ds2_name}_C",
            "Strict_Similarity_V",
            f"Approx_Similarity_{ds1_name}_V",
            f"Approx_Similarity_{ds2_name}_V",
            f"Inventory_{ds1_name}_Full",
            f"Inventory_{ds2_name}_Full",
            "Shared_Consonants",
            "Shared_Vowels",
            f"Exclusive_{ds1_name}_Consonants",
            f"Exclusive_{ds1_name}_Vowels",
            f"Exclusive_{ds2_name}_Consonants",
            f"Exclusive_{ds2_name}_Vowels",
        ]
        handler.write("\t".join(header))
        handler.write("\n")

        for glottocode in sorted(overlap):
            args.log.info("Comparing inventories for glottocode `%s`...", glottocode)
            lang1_id = lang1[glottocode]
            lang2_id = lang2[glottocode]
            inv1 = Inventory.from_list(*list(invs1[lang1_id]), clts=bipa)
            inv2 = Inventory.from_list(*list(invs2[lang2_id]), clts=bipa)

            similarity = {}
            for aspect_label in ["all", "consonant", "vowel", "tone"]:
                if aspect_label == "all":
                    aspects = ["consonant", "vowel", "tone"]
                else:
                    aspects = [aspect_label]

                try:
                    similarity[f"strict-{aspect_label}"] = "%.4f" % inv1.similar(
                        inv2, metric="strict", aspects=aspects
                    )
                except:
                    similarity[f"strict-{aspect_label}"] = ""

                try:
                    similarity[f"appr12-{aspect_label}"] = "%.4f" % inv1.similar(
                        inv2, metric="similarity", aspects=aspects
                    )
                except:
                    similarity[f"appr12-{aspect_label}"] = ""

                try:
                    similarity[f"appr21-{aspect_label}"] = "%.4f" % inv2.similar(
                        inv1, metric="similarity", aspects=aspects
                    )
                except:
                    similarity[f"appr21-{aspect_label}"] = ""

            # Collect consonants, vowels, and tones for both inventories
            # NOTE: already sorting here
            sounds1 = {
                aspect: sorted(list(inv1.sounds[aspect]))
                for aspect in ["consonant", "vowel", "tone"]
            }
            sounds2 = {
                aspect: sorted(list(inv2.sounds[aspect]))
                for aspect in ["consonant", "vowel", "tone"]
            }

            # get counts
            inv1_cons = len(inv1.sounds["consonant"])
            inv1_vowl = len(inv1.sounds["vowel"])
            inv2_cons = len(inv2.sounds["consonant"])
            inv2_vowl = len(inv2.sounds["vowel"])

            # build buffer
            buf = [
                glottocode,
                glottolog.languoid(glottocode).name,
                str(inv1_cons + inv1_vowl),
                str(inv2_cons + inv2_vowl),
                str(inv1_cons),
                str(inv2_cons),
                str(inv1_vowl),
                str(inv2_vowl),
                similarity["strict-all"],
                similarity["appr12-all"],
                similarity["appr21-all"],
                similarity["strict-consonant"],
                similarity["appr12-consonant"],
                similarity["appr21-consonant"],
                similarity["strict-vowel"],
                similarity["appr12-vowel"],
                similarity["appr21-vowel"],
                " ".join(sounds1["consonant"] + sounds1["vowel"]),
                " ".join(sounds2["consonant"] + sounds2["vowel"]),
                " ".join(
                    [snd for snd in sounds1["consonant"] if snd in sounds2["consonant"]]
                ),
                " ".join([snd for snd in sounds1["vowel"] if snd in sounds2["vowel"]]),
                " ".join(
                    [
                        snd
                        for snd in sounds1["consonant"]
                        if snd not in sounds2["consonant"]
                    ]
                ),
                " ".join(
                    [snd for snd in sounds1["vowel"] if snd not in sounds2["vowel"]]
                ),
                " ".join(
                    [
                        snd
                        for snd in sounds2["consonant"]
                        if snd not in sounds1["consonant"]
                    ]
                ),
                " ".join(
                    [snd for snd in sounds2["vowel"] if snd not in sounds1["vowel"]]
                ),
            ]

            handler.write("\t".join(buf))
            handler.write("\n")
