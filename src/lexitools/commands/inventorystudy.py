"""
Run a phonological inventory study.
"""

# Import Python standard libraries
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
import unicodedata
import random

# Import 3rd party libraries
import numpy as np
import powerlaw
import matplotlib.pyplot as plt
import scipy.stats

# Import MPI-SHH libraries
from clldutils.clilib import PathType
from cldfbench.cli_util import add_catalog_spec, get_dataset
from pylexibank.cli_util import add_dataset_spec

from lingpy.sequence.sound_classes import syllabify

# Silence `powerlaw` divide by zero warnings
np.seterr(divide="ignore", invalid="ignore")

SAMPLE_SIZE = [0.05, 0.25, 0.5, 0.666, 0.75, 0.90, 1.00]


def analyze_inventory(inventory, language, args):
    """
    Analyze an inventory, generating visualizations and returning data.
    """

    # Build array of frequencies
    data = np.array(sorted(list(inventory.values()), reverse=True))

    # Run powerlaw fitting (which includes other distributions);
    # data is discrete and we set the xmax to twice
    # the maximum observed value
    fit = powerlaw.Fit(data, discrete=True, xmax=max(data) * 2)

    # Collect data, first running distirbution comparisons
    pl_tpl_R, pl_tpl_p = fit.distribution_compare(
        "power_law", "truncated_power_law", normalized_ratio=True
    )
    pl_ln_R, pl_ln_p = fit.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True
    )
    pl_exp_R, pl_exp_p = fit.distribution_compare(
        "power_law", "exponential", normalized_ratio=True
    )
    pl_sexp_R, pl_sexp_p = fit.distribution_compare(
        "power_law", "stretched_exponential", normalized_ratio=True
    )
    tpl_ln_R, tpl_ln_p = fit.distribution_compare(
        "truncated_power_law", "lognormal", normalized_ratio=True
    )
    tpl_exp_R, tpl_exp_p = fit.distribution_compare(
        "truncated_power_law", "exponential", normalized_ratio=True
    )
    tpl_sexp_R, tpl_sexp_p = fit.distribution_compare(
        "truncated_power_law", "stretched_exponential", normalized_ratio=True
    )
    ln_exp_R, ln_exp_p = fit.distribution_compare(
        "lognormal", "exponential", normalized_ratio=True
    )
    ln_sexp_R, ln_sexp_p = fit.distribution_compare(
        "lognormal", "stretched_exponential", normalized_ratio=True
    )
    exp_sexp_R, exp_sexp_p = fit.distribution_compare(
        "exponential", "stretched_exponential", normalized_ratio=True
    )

    # Compute reference R values of each distribution for plot
    pl_R = (+pl_tpl_R + pl_ln_R + pl_exp_R + pl_sexp_R) / 4.0
    tpl_R = (-pl_tpl_R + tpl_ln_R + tpl_exp_R + tpl_sexp_R) / 4.0
    ln_R = (-pl_ln_R - tpl_ln_R + ln_exp_R + ln_sexp_R) / 4.0
    exp_R = (-pl_exp_R - tpl_exp_R - ln_exp_R + exp_sexp_R) / 4.0
    sexp_R = (-pl_sexp_R - tpl_sexp_R - ln_sexp_R - exp_sexp_R) / 4.0

    stats = {
        "alpha": fit.alpha,
        "D": fit.D,
        "xmin": fit.xmin,
        "power_law.alpha": fit.power_law.alpha,
        "power_law.sigma": fit.power_law.sigma,
        "truncated_power_law.alpha": fit.truncated_power_law.alpha,
        "truncated_power_law.lambda": fit.truncated_power_law.parameter2,
        "lognormal.mu": fit.lognormal.mu,
        "lognormal.sigma": fit.lognormal.sigma,
        "exponential.lambda": fit.exponential.parameter1,
        "stretched_exponential.lambda": fit.stretched_exponential.parameter1,
        "stretched_exponential.beta": fit.stretched_exponential.beta,
        "power_law-truncated_power_law-R": pl_tpl_R,
        "power_law-truncated_power_law-p": pl_tpl_p,
        "power_law-lognormal-R": pl_ln_R,
        "power_law-lognormal-p": pl_ln_p,
        "power_law-exponential-R": pl_exp_R,
        "power_law-exponential-p": pl_exp_p,
        "power_law-stretched_exponential-R": pl_sexp_R,
        "power_law-stretched_exponential-p": pl_sexp_p,
        "truncated_power_law-lognormal-R": tpl_ln_R,
        "truncated_power_law-lognormal-p": tpl_ln_p,
        "truncated_power_law-exponential-R": tpl_exp_R,
        "truncated_power_law-exponential-p": tpl_exp_p,
        "truncated_power_law-stretched_exponential-R": tpl_sexp_R,
        "truncated_power_law-stretched_exponential-p": tpl_sexp_p,
        "lognormal-exponential-R": ln_exp_R,
        "lognormal-exponential-p": ln_exp_p,
        "lognormal-stretched_exponential-R": ln_sexp_R,
        "lognormal-stretched_exponential-p": ln_sexp_p,
        "exponential-stretched_exponential-R": exp_sexp_R,
        "exponential-stretched_exponential-p": exp_sexp_p,
        "powerlaw.R": pl_R,
        "truncated_power_law.R": tpl_R,
        "lognormal.R": ln_R,
        "exponential.R": exp_R,
        "stretched_exponential.R": sexp_R,
    }

    # PDF/CCDF figure
    plt.clf()
    Fig_PDF_CCDF = fit.plot_pdf(color="b", linewidth=3)
    fit.power_law.plot_pdf(color="b", linestyle="--", label="PDF", ax=Fig_PDF_CCDF)
    fit.plot_ccdf(color="r", linewidth=3, ax=Fig_PDF_CCDF)
    fit.power_law.plot_ccdf(color="r", linestyle="--", label="CCDF", ax=Fig_PDF_CCDF)
    Fig_PDF_CCDF.set_ylabel(u"p(X),  p(X≥x)")
    Fig_PDF_CCDF.set_xlabel(r"%s Phoneme PDF and CCDF" % language)
    handles, labels = Fig_PDF_CCDF.get_legend_handles_labels()
    leg = Fig_PDF_CCDF.legend(handles, labels, loc=3)
    leg.draw_frame(False)
    save_figure(Fig_PDF_CCDF, "Fig_PDF_CCDF-%s" % language, args)

    # fitting figure
    plt.clf()
    Fig_Fits = fit.plot_ccdf(linewidth=3, label="Empirical Data")
    fit.power_law.plot_ccdf(
        ax=Fig_Fits, color="r", linestyle="--", label="Power law fit (R=%.2f)" % pl_R
    )
    fit.truncated_power_law.plot_ccdf(
        ax=Fig_Fits,
        color="g",
        linestyle="--",
        label="Truncated Power Law fit (R=%.2f)" % tpl_R,
    )
    fit.lognormal.plot_ccdf(
        ax=Fig_Fits, color="m", linestyle="--", label="Lognormal fit (R=%.2f)" % ln_R
    )
    fit.exponential.plot_ccdf(
        ax=Fig_Fits,
        color="lime",
        linestyle="--",
        label="Exponential fit (R=%.2f)" % exp_R,
    )
    fit.stretched_exponential.plot_ccdf(
        ax=Fig_Fits,
        color="yellow",
        linestyle="--",
        label="Stretched Exponential fit (R=%.2f)" % sexp_R,
    )
    Fig_Fits.set_ylabel(u"p(X≥x)")
    Fig_Fits.set_xlabel(r"%s Phoneme Frequency" % language)
    handles, labels = Fig_Fits.get_legend_handles_labels()
    Fig_Fits.legend(handles, labels, loc=3)
    save_figure(Fig_Fits, "Fig_Fits-%s" % language, args)

    return stats


def save_figure(figure, filename, args, format="png", dpi=150):
    """
    Writes figure to image file.
    """

    # Extract figure from matplotlib object
    out_figure = figure.get_figure()

    # build filename and write image
    filename = "%s.%s" % (filename, format)
    output_path = Path(args.output_dir) / filename
    out_figure.savefig(output_path.as_posix(), bbox_inches="tight", dpi=dpi)


def output_sample_stats(ks_stats, size_stats, phoneme_count, args):
    # Get languages
    langs = sorted(set([key[0] for key in ks_stats]))

    headers = ["Language", "Segments"]
    for sample_size in SAMPLE_SIZE:
        headers.append("KS_%.2f" % sample_size)
        headers.append("SIZE_%.2f" % sample_size)

    output_path = Path(args.output_dir) / "sample_stats.tsv"
    with open(output_path.as_posix(), "w") as handler:
        handler.write("\t".join(headers))
        handler.write("\n")

        for lang in langs:
            buf = [lang, str(len(phoneme_count[lang]))]
            for sample_size in SAMPLE_SIZE:
                buf.append("%.4f" % ks_stats[lang, sample_size])
                buf.append("%.2f" % size_stats[lang, sample_size])

            handler.write("\t".join(buf))
            handler.write("\n")


def output_powerlaw_stats(stats, args):

    fields = [
        "alpha",
        "D",
        "xmin",
        "power_law.alpha",
        "power_law.sigma",
        "truncated_power_law.alpha",
        "truncated_power_law.lambda",
        "lognormal.mu",
        "lognormal.sigma",
        "exponential.lambda",
        "stretched_exponential.lambda",
        "stretched_exponential.beta",
        "power_law-truncated_power_law-R",
        "power_law-truncated_power_law-p",
        "power_law-lognormal-R",
        "power_law-lognormal-p",
        "power_law-exponential-R",
        "power_law-exponential-p",
        "power_law-stretched_exponential-R",
        "power_law-stretched_exponential-p",
        "truncated_power_law-lognormal-R",
        "truncated_power_law-lognormal-p",
        "truncated_power_law-exponential-R",
        "truncated_power_law-exponential-p",
        "truncated_power_law-stretched_exponential-R",
        "truncated_power_law-stretched_exponential-p",
        "lognormal-exponential-R",
        "lognormal-exponential-p",
        "lognormal-stretched_exponential-R",
        "lognormal-stretched_exponential-p",
        "exponential-stretched_exponential-R",
        "exponential-stretched_exponential-p",
        "powerlaw.R",
        "truncated_power_law.R",
        "lognormal.R",
        "exponential.R",
        "stretched_exponential.R",
    ]

    output_path = Path(args.output_dir) / "pl_stats.tsv"
    with open(output_path.as_posix(), "w") as handler:
        # write headers
        headers = ["LANGUAGE"] + [header.upper() for header in fields]
        handler.write("\t".join(headers))
        handler.write("\n")

        # Write language data
        for language in sorted(stats):
            # build the buffer with values
            buf = [language] + [stats[language][field] for field in fields]

            # get a string representation of all fields
            buf = [value if isinstance(value, str) else "%.4f" % value for value in buf]

            # Write
            handler.write("\t".join(buf))
            handler.write("\n")


def normalize_grapheme(grapheme, bipa):
    """
    Normalizes a grapheme.

    Does Unicode normalization, splitting in case of Phoible/slash-notation,
    BIPA default selection.
    """

    # Unicode normalization
    grapheme = unicodedata.normalize("NFC", grapheme)

    # Split over slash notation, if any, keeping the entry to the right
    if "/" in grapheme:
        grapheme = grapheme.split("/")[1]

    # Only split the vertical bar, as used in Phoible, if the grapheme is
    # longer than one character (can be a click), keeping the first one
    if len(grapheme) > 1 and "|" in grapheme:
        grapheme = grapheme.split("|")[0]

    # Normalize BIPA
    grapheme = str(bipa[grapheme])

    return grapheme


# TODO: add flag for stripping tones
def lexeme2phonemes(lexeme, bipa, **kwargs):
    """
    Given a segment string, return it as a list of phonemes.

    Parameters
    ----------
    bipa : BIPA object
        CLTS object for grapheme normalization
    keep_mark : bool
        Whether to keep markers (default: remove)
    """

    # Parse flags for phoneme processing
    keep_mark = kwargs.get("keep_mark", False)

    # Perform all processing
    phonemes = [normalize_grapheme(phoneme, bipa) for phoneme in lexeme.split()]

    if not keep_mark:
        phonemes = [phoneme for phoneme in phonemes if phoneme not in ["+", "_"]]

    return phonemes


# TODO: add flag for stripping tones
# TODO: parse as onset, nucleus, coda
def lexeme2syllables(lexeme):

    # Process the lexeme; the syllabification algorithm is stricter,
    # so we cannot have the option of keeping slashes, marks, etc
    phonemes = [
        phoneme.split("/")[1] if "/" in phoneme else phoneme
        for phoneme in lexeme.split()
    ]
    phonemes = [phoneme for phoneme in phonemes if phoneme not in ["+", "_"]]
    clean_lexeme = " ".join(phonemes)

    # Get the syllables
    syllables = syllabify(clean_lexeme, output="nested")

    return syllables


def read_extended_data(ds, args):
    """
    Reads phonemes and syllables from a dataset.
    """

    # Collect data; for the time being only Language_ID (for grouping)
    # and Segments. No other variable is taken into account, particularly
    # the concept, which in the future could be used for matters like
    # phonotactics
    data = []
    for row in ds.cldf_dir.read_csv("forms.csv", dicts=True):
        phonemes = lexeme2phonemes(row["Segments"], args.clts.api.bipa)
        syllables = lexeme2syllables(row["Segments"])

        data.append(
            {
                "Language_ID": row["Language_ID"],
                "Segments": row["Segments"],
                "Phonemes": phonemes,
                "Phoneme_Length": len(phonemes),
                "Syllables": syllables,
                "Syllable_Length": len(syllables),
            }
        )

    return data


def collect_inventories(data):
    """
    Collect inventories per language.
    """

    vocabularies = defaultdict(list)
    syllabaries = defaultdict(list)
    for entry in data:
        vocabularies[entry["Language_ID"]].append(entry["Phonemes"])
        syllabaries[entry["Language_ID"]] += [
            " ".join(syllable) for syllable in entry["Syllables"]
        ]

    # Collect phoneme and syllable counters per language
    phoneme_count = {
        language: Counter(chain.from_iterable(lexemes))
        for language, lexemes in vocabularies.items()
    }
    syllable_count = {
        language: Counter(syllables) for language, syllables in syllabaries.items()
    }

    return phoneme_count, syllable_count


def collect_sampled_inventories(data, k=10):
    """
    Collect inventories per language.
    """

    random.seed()

    vocabularies = defaultdict(list)
    for entry in data:
        vocabularies[entry["Language_ID"]].append(entry["Phonemes"])

    counts = defaultdict(dict)
    for lang, vocab in vocabularies.items():
        for sample_size in SAMPLE_SIZE:
            counts[sample_size][lang] = []
            size = int(len(vocab) * sample_size)
            for i in range(k):
                sample = random.sample(vocab, size)
                phoneme_count = Counter(chain.from_iterable(sample))
                counts[sample_size][lang].append(phoneme_count)

    return counts


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
    parser.add_argument("output_dir", type=str, help="Path to the output directory")


def run(args):
    """
    Entry point for command-line call.
    """

    # Extract dataset
    ds = get_dataset(args)

    # Read raw data and extend it with phonological information
    args.log.info("Loading data from %s...", ds)
    data = read_extended_data(ds, args)
    args.log.info("Read %i entries from CLDF.", len(data))

    # Collect inventories
    args.log.info("Collecting inventories...")
    phoneme_count, syllable_count = collect_inventories(data)
    args.log.info("Read %i inventories.", len(phoneme_count))

    # Collect inventories by size, testing the sample size needed
    args.log.info("Estimating sample sizes...")
    sampled = collect_sampled_inventories(data)
    args.log.info("Read %i inventories.", len(sampled))

    # Estimate sample sizes, and compute the means for output
    ks_stats = defaultdict(list)
    size_stats = defaultdict(list)
    for lang, full in phoneme_count.items():
        dist1 = [full.get(sound, None) for sound in sorted(full)]
        for sample_size in sampled:
            for i, sample in enumerate(sampled[sample_size][lang]):
                dist2 = [sample.get(sound, 0) for sound in sorted(full)]

                ks, p = scipy.stats.ks_2samp(dist1, dist2)

                ks_stats[lang, sample_size].append(ks)
                size_stats[lang, sample_size].append(len(sample))

    ks_stats = {key: np.mean(ks_values) for key, ks_values in ks_stats.items()}
    size_stats = {
        key: np.mean([[size / len(phoneme_count[key[0]])] for size in sizes])
        for key, sizes in size_stats.items()
    }

    output_sample_stats(ks_stats, size_stats, phoneme_count, args)

    # iterate over all phoneme inventories
    stats = {}
    for language, inventory in phoneme_count.items():
        args.log.info("Processing inventory for %s...", language)
        lang_stats = analyze_inventory(inventory, language, args)
        stats[language] = lang_stats

    # Output statistics
    args.log.info("Writing results...")
    output_powerlaw_stats(stats, args)
