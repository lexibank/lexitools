"""
check structure if the tokens are matching the template.
"""
from clldutils.clilib import PathType
from cldfbench.cli_util import get_dataset
from tabulate import tabulate
from linse.annotate import seallable
from linse.transform import morphemes


def get_structure(sequence):
    """
    produce a list of structure tokens 
    """
    out = []
    for m in morphemes(sequence):
        out += [
            "".join(
                seallable(
                    m,
                    medials={
                        "j",
                        "w",
                        "jw",
                        "wj",
                        "i̯",
                        "u̯",
                        "i̯u̯",
                        "u̯i̯",
                        "iu",
                        "ui",
                        "y",
                        "ɥ",
                        "l",
                        "lj",
                        "lʲ",
                        "r",
                        "rj",
                        "rʲ",
                        "ʐ",
                        "ʑ",
                        "ʂ",
                        "ʂ",
                        "rʷ",
                        "lʷ",
                    },
                )
            )
        ]
    return list("+".join(out))


def run(args):
    """
    main function.
    """
    ds = get_dataset(args)
    df = {}
    # ds =Dataset()
    #  work only if the forms.csv exists
    if ds.cldf_dir.joinpath("forms.csv").exists():
        for row in ds.cldf_reader()["FormTable"]:
            i = row["ID"]
            form = row["Form"]
            segment = row["Segments"]
            doculect = row["Language_ID"]
            concept = row["Parameter_ID"]
            # illegal ending
            if str(segment).endswith("+") or str(segment).startswith("+"):
                print("{0}\t{1}\t{2}\t{3}".format("illegal ending", i, form, segment))
            # empty morpheme
            elif "+ +" in str(segment):
                print("{0}\t{1}\t{2}\t{3}".format("empty morpheme", i, form, segment))
            else:
                df[i] = [form, segment, get_structure(segment), doculect, concept]

    # checking the length between segments and structure
    errors = []
    for key, value in df.items():
        error = ""
        if len(value[1]) != len(value[2]):
            error = "wrong length"
        elif not "n" in value[2]:
            error = "missing vowel"

        if error.strip():
            errors += [[key, value[3], value[4], value[0], value[1], value[2], error]]
    table = sorted(errors, key=lambda x: (x[-1], x[-2], x[1]))
    for i, line in enumerate(table):
        table[i] = [i + 1] + line
    print(
        tabulate(
            table,
            headers=[
                "Count",
                "ID",
                "Doculect",
                "Concept",
                "Form",
                "Token",
                "Structure",
                "Error",
            ],
            tablefmt="pipe",
        )
    )


morphemes = set([(line[-4], str(line[-3]), str(line[-2])) for line in table])
for a, b, c in sorted(morphemes, key=lambda x: x[-2]):
    print(a + "\t" + b + "\t" + c)
