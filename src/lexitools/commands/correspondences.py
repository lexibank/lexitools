"""
Extract correspondences from a set of datasets.
"""
from clldutils.clilib import add_format
from cldfbench.cli_util import add_catalog_spec
from pyconcepticon.api import Concepticon
from cldfcatalog import Config
import lingpy
import lingpy.evaluate
from lingpy.compare import partial
from lingpy.compare.util import mutual_coverage_check
from lingpy.compare.sanity import average_coverage
from collections import Counter, defaultdict, namedtuple
from pylexibank import progressbar
import pyglottolog
import csv
import subprocess
from importlib.util import find_spec
from cldfbench import get_dataset
import sys
import json
from lexitools.coarse_soundclass import Coarsen
import time
import logging
import lingrex
from pathlib import Path

# Do not merge consecutive vowels into diphthongs
lingpy.settings.rcParams["merge_vowels"] = False

# import cProfile
from memory_profiler import profile

# TODO: rewrite to work in two steps:
# 1. export a wordlist per family with all useful info
# 2. run on a single family (this allows for simple parallelisation)


# These ones have cognate ids:
# bdpa
# birchallchapacuran
# lieberherrkhobwa
# sagartst
# walworthpolynesian
# yanglalo
# baf2
# starostinhmongmien
# starostinkaren
# wichmannmixezoquean

# TODO: synonyms: prune identical rows which differ only by dataset


# set of characters which should be ignored in tokens (markers, etc)
IGNORE = {"+", "*", "#", "_", ""}

## Back to namedtuples which are more memory efficient and enough for our purposes

"""A Language has a name, a glottocode, and a family."""
Lang = namedtuple("Lang", ('glottocode', 'family', 'name'))

class MagicGap(str):
    """ A gap with context, which Lingrex recognizes as a gap.

        In Lingrex: `x == '-'` is used to check if a token is a gap.
        We need symbols that pass that test, but which carry contextual information,
            and are different from gaps in different contexts.
    """
    def __new__(cls, left, right, *args, **kwargs):
        s = str.__new__(cls, "-")
        s.left = left
        s.right = right
        return s
    def __ne__(self, other): return not self == other
    def __eq__(self, other):
        if type(other) is MagicGap: return str(self) == str(other)
        return other == "-"
    def __str__(self): return self.left + "|-|" + self.right
    def __hash__(self): return hash("-")

class MockLexicore(object):
    """ Mock interface to lexicore datasets.

    This loads a list of datasets. If not installed, it tries to pip install them,
    then the script needs to be re-ran. This is a temporary fix in order to use Lexicore
    before it really exists. The interface was agreed upon so that we can plug  in the
    correct interface soon with little effort.

    Attributes:
        datasets (dict of str to cldfbench.Dataset): maps dataset names to Dataset objects.
    """

    def __init__(self, dataset_path):
        """ Load all datasets from a list of dataset indentifiers, or install them.

        Note: this does NOT update installed datasets which are out of date.

        Args:
            dataset_path (list): path to a table with datasets to download.
        """

        self.dataset_list = []

        with dataset_path.open("r") as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            next(csvReader)  # get rid of the header
            for row in csvReader:
                self.dataset_list.append((row[1], row[2]))
        #
        # dataset_list = [
        #
        #     ('lexibank', 'allenbai'),
        #     ('lexibank', 'sagartst'),
        #     ('lexibank', 'starostinkaren'),
        #     ('lexibank', 'bdpa'),
        #     ('lexibank', 'birchallchapacuran'),
        #     ('lexibank', 'lieberherrkhobwa'),
        #     ('lexibank', 'walworthpolynesian'),
        #     ('lexibank', 'yanglalo'),
        #     ('lexibank', 'baf2'),
        #     ('lexibank', 'wichmannmixezoquean'),
        #     ('lexibank', 'starostinhmongmien'),
        # ]

        self.download_dataset()

        self.datasets = {}
        for org, name in self.dataset_list:
            try:
                self.datasets[name] = get_dataset(name,
                                                  ep="lexibank.dataset").cldf_reader()
            except AttributeError as e:
                print("Failed to load", name)
                print(e)
        if not self.datasets:
            raise Exception("No dataset loaded")

    def download_dataset(self):
        # see https://github.com/chrzyki/snippets/blob/main/lexibank/install_datasets.py
        egg_pattern = "git+https://github.com/{org}/{name}.git#egg=lexibank_{name}"
        successful_install = []
        failed_install = []
        for org, name in self.dataset_list:
            if find_spec("lexibank_" + name) is None:
                args = [sys.executable, "-m", "pip", "install",
                        "-e", egg_pattern.format(org=org, name=name)]
                res = subprocess.run(args)
                if res.returncode == 0:
                    successful_install.append(org + "/" + name)
                else:
                    failed_install.append(org + "/" + name)
        if successful_install or failed_install:
            msg = ["Some datasets were not installed."]
            if successful_install:
                msg.extend(["I installed:", ", ".join(successful_install), "."])
            if failed_install:
                msg.extend(["I failed to install: ", " ".join(failed_install), "."])
            msg.extend(["Please install any missing datasets and re-run the command."])
            raise EnvironmentError(" ".join(msg))

    def iter_table(self, table_name):
        """ Iter on the rows of a specific table, across all loaded datasets.

        Args:
            table_name (str): name of a CLDF Wordlist table.

        Yields:
            rows from all datasets, with an additional "dataset" field indicating the name
            of the dataset from which the row originates.
        """
        for name, ds in self.datasets.items():
            for row in ds[table_name]:
                row["dataset"] = name
                yield row

    def get_table(self, table_name):
        """ Return all the rows of a specific tablee, across all loaded datasets.

        Args:
            table_name (str): name of a CLDF Wordlist table.

        Returns:
            list of rows
        """
        return list(self.iter_table(table_name))


def wordlist_subset(lex, f):
    out = {}
    for i, row in lex._data.items():
        if i == 0:
            out[i] = row
        if f(i, row):
            out[i] = row
    return out


class CorrespFinder(object):
    """ Prepare lexicore data, search and align cognates, find patterns.

    Attributes:
        errors (list of list): table which summarizes errors encountered in tokens.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of Lang to concepts.
        _data (defaultdict): mapping of (lang, concept) -> token string -> word instances
           A language can have more that one token for a specific
         concept, and a single token can be found in more than one dataset. A Word represents
         exactly one token in a given language and dataset.
    """

    def __init__(self, db, prefix, coarsen, glottolog, concept_list=None):
        """ Initialize the data by collecting, processing and organizing tokens.

        Iterates over the database, to:
        - Identify the family for all languages
        - If specified, restrict data to a specific list of concepts.
        - Pre-process tokens to replace sounds by sound classes and count syllables
        - Make note of any tokens which result in errors.

        Args:
            db (MockLexicore): Lexicore database
            concept_list (str): name of a concepticon concept list
            sound_class (func): a function which associates a class of sounds to any BIPA
            sound string.
            glottolog (pyglottolog.Glottolog): Glottolog instance to retrieve family names.
        """
        # This mixes some lingpy specific headers (used to init lexstat)
        # and some things we need specifically for this application
        namespace = ['doculect', 'concept', 'concepticon', 'original_token',
                     'original_id', 'tokens', 'structure',
                     'glottocode', 'cogid', 'cldf_dataset']
        self.cols = {n: i for i, n in enumerate(namespace)}
        self.evals = {}
        self.stats = {}
        self.prefix = prefix

        self.coarsen = coarsen
        #
        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        self.languages = self.prepare_languages(db, glottolog)

        self.errors = [["Dataset", "Language_ID", "Sound", "Token", "ID"]]

        concepts = {row["ID"]: (row["Concepticon_ID"], row["Concepticon_Gloss"])
                    for row in
                    db.iter_table('ParameterTable')}

        # Define concept list
        if concept_list is not None:
            concepticon = Concepticon(Config.from_file().get_clone("concepticon"))
            concept_dict = concepticon.conceptlists[concept_list].concepts
            self.concepts_subset = {c.concepticon_id for c in
                                    concept_dict.values() if
                                    c.concepticon_id}
        else:  # All observed concepts
            self.concepts_subset = {i for i, j in concepts.values()}

        # dict of family -> (language, concept, word) -> row
        self._data = defaultdict(lambda: [namespace])
        self.cogids = defaultdict(set)

        duplicates = Counter()

        for row in progressbar(db.iter_table('FormTable'),
                               desc="Loading data..."):

            # Ignore loan words
            if row.get("Loan", ""): continue

            # Ignore words without a concept in the current list,
            # with an incorrect language identifier, or without a form
            concept_id, concept_gloss = concepts[row["Parameter_ID"]]

            if concept_id is None or \
                    concept_id not in self.concepts_subset or \
                    row["Language_ID"] not in self.languages or \
                    row["Segments"] is None:
                continue

            # Convert sounds, ignore rows with unknown or ignored sounds
            try:
                tokens = list(self._iter_phonemes(row))
                structure = [t if t in IGNORE else "A"  for t in tokens]
            except ValueError:
                continue  # unknown sounds
            if all([s in IGNORE for s in tokens]): continue

            doculect = row["Language_ID"]
            lang = self.languages[doculect]
            dataset = row["dataset"]

            cogid = row.get("Cognacy", row.get("Cognateset_ID", None))

            self.cogids[(lang.family, dataset)].add(cogid)

            # skip any duplicates.
            ipa_word = tuple(tokens)
            if duplicates[(doculect, concept_id, ipa_word)] > 0: continue

            duplicates[(doculect, concept_id, ipa_word)] += 1

            # add row
            self._data[lang.family].append([doculect, concept_gloss, concept_id,
                                            " ".join(row["Segments"]), row["ID"],
                                            tokens, structure,
                                            lang.glottocode, cogid, dataset])

        # lingpy wordlist internal format
        self._data = {f: dict(enumerate(self._data[f])) for f in self._data}
        self.prepare_cogids()

    def prepare_cogids(self):
        # We need to convert all cogids to ints, and often they're unfortunately strings
        cog_to_int = {}
        next_cogid = 1

        def single_cogid(dataset, cogid):
            nonlocal next_cogid
            if (dataset, cogid) not in cog_to_int:
                cog_to_int[(dataset, cogid)] = next_cogid
                next_cogid += 1
            return cog_to_int[(dataset, cogid)]

        def multiple_cogids(dataset, cogids):
            return tuple(single_cogid(dataset, x) for x in cogids)

        # For each dataset, determine whether cogids are lists or ints
        partial = {}
        for family, dataset in self.cogids:
            partial[dataset] = any(c is not None and " " in c for c in self.cogids[(family, dataset)])


        # Convert cogids to either ints or lists of ints
        c = self.cols["cogid"]
        t = self.cols["original_token"]
        d = self.cols["cldf_dataset"]
        for family in self._data:
            for i, row in self._data[family].items():
                if i > 0:
                    cogid = row[c]
                    dataset = row[d]
                    if cogid in {"0", 0, None, ""}:
                        row[c] = None
                    elif partial[dataset]:
                        cogids = cogid.split(" ")
                        morphemes = row[t].split(" + ")
                        if len(cogids) != len(morphemes):
                            # Sometimes, I find a single cogid with a word which has multiple morphemes
                            # I think this happens because of words which have multiple
                            # morphemes but have no cognates in the dataset.
                            row[c] = tuple(single_cogid(dataset, cogid) for _ in range(len(morphemes)))
                        else:
                            row[c] = multiple_cogids(dataset, cogids)
                    else:
                        row[c] = single_cogid(dataset, cogid)


    def prepare_languages(self, db, glottolog):
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        languages = {}
        for row in progressbar(db.iter_table("LanguageTable"), desc="listing languages"):
            gcode = row["Glottocode"]
            if gcode is None: continue
            family = row["Family"]
            langoid = langoids.get(gcode, None)
            if langoid is not None and langoid.family is not None:
                family = langoid.family.name
            if family == "Isolate":
                family = langoid.name + "_isolate"

            languages[row["ID"]] = Lang(family=family, glottocode=gcode, name=row[
                "Name"])  # TODO: should we use the langoid name here ?
        return languages

    def sanity_stats(self, lex):
        def mut_cov():
            for i in range(lex.height, 1, -1):
                if mutual_coverage_check(lex, i):
                    return i
            return 0

        d = {"min_mutual_coverage": mut_cov()}

        try:
            d["average_coverage"] = average_coverage(lex)
        except ZeroDivisionError:
            d["average_coverage"] = 0
        return d

    def _find_cognates(self, family, pbar, eval, pred_params):
        # Define the current list of cognate ids (if any)
        # Decide whether we need partial or wholistic cognate detection
        if pred_params["segmented"]:  # Cognates and alignments are morpheme-level
            lex = partial.Partial(self._data[family], check=True)
        else:  # Cognates and alignments are word-level
            lex = lingpy.LexStat(self._data[family], check=True)

        if lex.height < 2 or lex.width < 2:
            datasets = {r[self.cols["cldf_dataset"]] for i, r in self._data[family].items() if i>0}
            print("Family %s has no usable data (datasets: %s)" % (
                family, ", ".join(datasets)))
            return None

        # Clear some memory...
        del self._data[family]

        self.stats[family] = self.sanity_stats(lex)

        # prediction needed when we don't have all cogids, or when evaluating
        if pred_params["needs_prediction"] or eval:
            # Decide which algorithm to run
            kw = dict(method='lexstat', threshold=0.55, ref="pred_cogid",
                      cluster_method='infomap')

            if self.stats[family]["min_mutual_coverage"] < 100:
                kw = dict(method="sca", threshold=0.45, ref='pred_cogid',
                          no_bscorer=True)

            # Run cognate detection
            if pred_params["segmented"]:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.stats[family]["detection_type"] = 'morpheme',
                self.stats[family]["cognate_source"] = "Partial(" + kw_str + ")"
                pbar.set_description("looking for partial cognates in family %s" % family)
                lex.get_scorer(runs=1000)
                lex.partial_cluster(**kw)
            else:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.stats[family]["detection_type"] = 'word',
                self.stats[family]["cognate_source"] = "LexStat(" + kw_str + ")"
                pbar.set_description("looking for cognates in family %s" % family)
                lex.get_scorer(runs=1000)
                lex.cluster(**kw)

        ## Evaluate cognate detection
        if eval:
            pbar.set_description("evaluating in %s" % family)
            self._eval_per_dataset(lex, family, pred_params)

        # Either we didn't predict, or we predicted only for evaluation purposes
        if not pred_params["needs_prediction"]:
            self.stats[family][
                "detection_type"] = 'morpheme' if pred_params["gold_segmented"] else 'word'
            lex.add_entries("pred_cogid", "cogid", lambda x: x, override=True)
            self.stats[family]["cognate_source"] = "expert"

        return lex

    def _eval_per_dataset(self, lex, family, pred_params):
        eval_measures = ["precision", "recall", "f-score"]
        # separate data by dataset, keep only if gold annotation exists
        columns = lex.columns
        by_datasets = defaultdict(lambda: [list(columns)])
        for i, r in lex._data.items():
            if i > 0 and r[lex._header['cogid']] is not None:
                dataset = r[lex._header['cldf_dataset']]
                by_datasets[dataset].append(
                    list(r))  # copy of rows, otherwise we edit lex

        # Evaluate inside each dataset
        for dataset in progressbar(list(by_datasets)):
            gold_rows = dict(enumerate(by_datasets[dataset]))

            eval_lex = lingpy.Wordlist(gold_rows)

            if eval_lex.height < 2 or eval_lex.width < 2:
                print(
                    "Dataset %s in family %s has no usable eval data" % (dataset, family))
                continue

            if pred_params["segmented"] and pred_params["gold_segmented"]:
                res = lingpy.evaluate.acd.partial_bcubes(eval_lex, gold='cogid',
                                                         test='pred_cogid',
                                                         pprint=False)
                # Diff won't work with lists, needs a tuple
                eval_lex.add_entries("pred_cogid", "pred_cogid",
                                     tuple,
                                     override=True)
            else:
                if pred_params["segmented"]:  # revert to word-level cognates
                    eval_lex.add_entries("pred_cogid", "pred_cogid",
                                         lambda x: " ".join(str(i) for i in x),
                                         override=True)

                res = lingpy.evaluate.acd.bcubes(eval_lex, gold='cogid',
                                                 test='pred_cogid',
                                                 pprint=False)
            lingpy.evaluate.acd.diff(eval_lex, 'cogid', 'pred_cogid', tofile=True,
                                     filename="{}_{}_{}_cognate_eval".format(
                                         self.prefix, family, dataset),
                                     pprint=False)

            d = dict(zip(eval_measures, res))
            d.update(self.sanity_stats(eval_lex))
            d["cognate_source"] = self.stats[family]["cognate_source"]
            d["cognates"] = len({cog for i, cog in eval_lex.iter_rows("cogid")})
            d["languages"] = eval_lex.width
            d["tokens"] = len(eval_lex)
            d["detection_type"] = "morpheme" if pred_params["segmented"] else "word"
            self.evals[(family, dataset)] = d
            del by_datasets[dataset]

    def _find_patterns(self, eval=False):


        def format_ex(alm, i):
            template = "{word} ([{original_token}] {cldf_dataset} {original_id})"
            return template.format(word=" ".join(alm[i, "tokens"]),
                                   original_token=alm[i, "original_token"],
                                   cldf_dataset=alm[i, "cldf_dataset"],
                                   original_id=alm[i, "original_id"])

        lingpy.log.get_logger().setLevel(logging.WARNING)
        pbar = progressbar(list(self._data),
                           desc="looking for cognates...")

        for family in pbar:

            pred_params = self._get_prediction_params(family)

            # Find cognates
            lex = self._find_cognates(family, pbar, eval, pred_params)

            if lex is None: continue # skip this family

            # Align cognates
            alm = lingpy.Alignments(lex,
                                    ref="pred_cogid", fuzzy=pred_params["partial"])
            del lex # trying to free more memory...
            pbar.set_description("Aligning cognates in %s" % family)
            alm.align(method='library', iteration=True, model="sca", mode="global")

            # Override alignment to add contexts (no loss of info)
            pbar.set_description("Adding contexts in %s" % family)
            msa = alm.msa["pred_cogid"]
            for cogid in msa:
                msa[cogid]["alignment"] = [list(self.add_contexts(seq)) for seq in msa[cogid]["alignment"]]

            # Consolidate alignment sites into patterns
            pbar.set_description("Clustering sites in patterns in %s" % family)
            cp = lingrex.CoPaR(alm, ref="pred_cogid", fuzzy=pred_params["partial"],
                                  segments="tokens",  structure="structure")
            del alm # trying to free more memory...
            cp.get_sites()
            cp.cluster_sites()

            langs = cp.cols

            pbar.set_description("Preparing patterns in %s" % family)

            # print(langs)
            # Iterate over patterns
            for pat_idx, ((_, pattern), sites) in enumerate(cp.clusters.items()):
                # Unique identifier for this pattern
                pattern_id = family + "-" + str(pat_idx)
                # print(pattern)

                l = sum([x != 'Ø' for x in pattern]) # Number of sounds in the pattern
                site_count = len(sites) # Number of sites displaying the pattern
                sites_refs = [dict(zip(cp.msa["pred_cogid"][cog]["taxa"],
                                       cp.msa["pred_cogid"][cog]["ID"])) for cog, pos in sites]
                # print([cp.msa["pred_cogid"][cog] for cog, pos in sites])

                # yield one row for each sound in the pattern
                for i in range(len(pattern)):
                    lang = langs[i]
                    sound = str(pattern[i]) # if it's a magicGap, this converts it to a context string

                    if sound == 'Ø' or sound in IGNORE: continue

                    # Format examples
                    examples = []
                    for ref in sites_refs:
                        if lang in ref: # If this alignment had this language
                            examples.append(format_ex(cp, ref[lang]))
                        if len(examples) == 3: break


                    yield [family, pattern_id, site_count, *sound.split("|"),
                           self.languages[lang].name,
                           self.languages[lang].glottocode,
                           ";".join(examples)]
            del cp
            # TODO: notes for Erich: having this context specific makes it a little sparser, might want to merge them.


    def _get_prediction_params(self, family):
        cogids_counts = Counter()
        segmented = False
        gold_segmented = False
        for i, row in self._data[family].items():
            cogid = row[self.cols["cogid"]]
            if "+" in row[self.cols["tokens"]]:
                segmented = True
            if type(cogid) is tuple:
                gold_segmented = True
            cogids_counts[cogid] += 1
        needs_prediction = None in cogids_counts
        return {"gold_segmented":gold_segmented,
                "needs_prediction": needs_prediction,
                "partial": (needs_prediction and segmented) or (gold_segmented),
                "segmented": segmented}


    def add_contexts(self, seq):
        """ Iterator of sounds and contexts for a pair of aligned tokens.

        Args:
            seq (list of str): sequence of segments or gap.

        Yields: pair of aligned sounds and their contexts: `(sound, context)`
        """

        def to_categories(sequence):
            """Turn a sequence of sounds into a sequence of categories used in contexts"""
            for s in sequence:
                if s == "-":
                    yield None
                elif s in IGNORE:
                    yield s
                else:
                    cat = self.coarsen.category(s)
                    if cat in {"T"}:
                        yield None
                    else:
                        yield cat
            yield "#"

        def get_right_context(cats, i):
            """Return the context for a given sound."""
            return next((c for c in cats[i + 1:] if c is not None))

        cats = list(to_categories(seq))
        l = len(seq)
        left = "#"
        for i in range(l):
            right = get_right_context(cats, i)
            if seq[i] == "-": # output a gap which retains context info
                yield MagicGap(left, right)
            elif seq[i] in IGNORE: # output markers as is
                yield seq[i]
            else:
                yield left+"|"+seq[i]+"|"+ right
            left = left if cats[i] is None else cats[i]

    def _iter_phonemes(self, row):
        """ Iterate over pre-processed phonemes from a row's token.

        The phonemes are taken from the "Segments" column

        We pre-process by:
            - re-tokenizing on spaces, ignoring empty segments
            - selecting the second element when there is a slash
            - using the sound_class attribute function to obtain sound classes

        Args:
            row (dict): dict of column name to value

        Yields:
            successive sound classes in the row's word.
        """
        segments = row["Segments"]
        # In some dataset, the separator defined in the metadata is " + ",
        # which means that tokens are not phonemes (ex:bodtkhobwa)
        # This is solved by re-tokenizing on the space...
        tokens = " ".join([s for s in segments if (s is not None and s != "")]).split(" ")
        l = len(tokens)
        for i, segment in enumerate(tokens):
            try:
                if "/" in segment:
                    segment = segment.split("/")[1]
                segment = self.coarsen[segment]
                if (
                        i == 0 or i == l - 1) and segment == "+":  # ignore initial and final "+"
                    continue
                yield segment
            except ValueError as e:
                self.errors.append((row["dataset"], row["Language_ID"], segment,
                                    " ".join(str(x) for x in segments), row["ID"]))
                raise e


def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")
    add_catalog_spec(parser, "glottolog")
    add_format(parser, default='pipe')

    parser.description = run.__doc__

    parser.add_argument(
        '--display',
        action='store',
        default=None,
        help='select a display')

    #
    # parser.add_argument(
    #     '--alignment',
    #     choices=["sca", "simple"],
    #     default='simple',
    #     type=str,
    #     help='select an alignment method: either of SCA or a simple scorer '
    #          'which penalizes C/V matches and forbids T/C & T/V matches.')
    parser.add_argument(
        '--cognate_eval',
        action='store_true',
        default=False,
        help='Evaluate cognate detection.')

    parser.add_argument(
        '--concepts',
        action='store',
        default=None,
        type=str,
        help='select a concept list to filter on')

def run(args):
    """Run the correspondence command.

    Run with:

        cldfbench lexitools.correspondences --clts-version v1.4.1 --dataset lexicore

    For details on the arguments, see `cldfbench lexitools.correspondences --help`.

    This loads all the requested datasets and searches for available sounds and attested
    correspondences. It output a series of files which start by a time-stamp,
    then "_sound_correspondences_" and end in:

    `_coarsening.csv`: output only if run with the model "Coarse". This is a table of all
        known coarse sounds.
    `_counts.csv`: a csv table recording correspondences. The header row is:
    `family,lang_a,lang_b,sound_a,sound_b,env_a,env_b,count`.
        The order of languages A/B and sounds A/B is not meaningful,
        we do not duplicate A/B and B/A. Env A and B are the environments in which sounds
        were observed (their contexts).
    `_sound_errors.csv`: records sounds that raised errors in CLTS. The corresponding
        tokens have been ignored by the program. The header row is
        `Dataset,Language_ID,Sound,Token,ID`.
    `_metadata.json`: a json file recording the input parameters and all relevant metadata.
    """
    args.log.info(args)
    clts = args.clts.from_config().api

    now = time.strftime("%Y%m%d-%Hh%Mm%Ss")
    output_prefix = "{timestamp}_sound_correspondences".format(timestamp=now)

    coarsening_file = (Path(__file__) / "../../../../etc/default_coarsening.csv").resolve()
    coarse = Coarsen(clts.bipa, coarsening_file)

    ## This is a temporary fake "lexicore" interface
    dataset_path =  (Path(__file__) / '../../../../etc/lexicore_list.csv').resolve()

    dataset = MockLexicore(dataset_path)
    corresp_finder = CorrespFinder(dataset,
                         output_prefix, coarse,
                         pyglottolog.Glottolog(args.glottolog.dir),
                         concept_list=args.concepts)

    args.log.info(
        'Loaded the wordlist ({} languages, {} families, {} concepts kept)'.format(
            len(corresp_finder.languages),
            len(corresp_finder._data),
            len(corresp_finder.concepts_subset)))


    # with cProfile.Profile() as pr:


    # Find all correspondences and write
    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(
            ["family", "pat_id", "count", "left_context", "sound", "right_context",
             "language", "glottocode", "examples"])
        writer.writerows(corresp_finder._find_patterns(args.cognate_eval))

    # pr.dump_stats("profile.prof")

    with open(output_prefix + '_coarsening.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerows(coarse.as_table())


    with open(output_prefix + '_cognate_info.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(
            ["family", "min_mutual_coverage", "average_coverage", "cognate_source",
             "cognate_type"])
        for family in corresp_finder.stats:
            infos = corresp_finder.stats[family]
            writer.writerow(
                [family, infos["min_mutual_coverage"], infos["average_coverage"],
                 infos["cognate_source"], infos["detection_type"]])

    metadata_dict = {"concepts": args.concepts,
                     "dataset": args.dataset}
    dataset_str = sorted(a + "/" + b for a, b in dataset.dataset_list)
    metadata_dict["dataset_list"] = dataset_str
    # metadata_dict["n_languages"] = len(data.lang_to_concept)
    metadata_dict["n_families"] = len(corresp_finder._data)
    metadata_dict["n_concepts"] = len(corresp_finder.concepts_subset)
    metadata_dict["n_tokens"] = sum([len(corresp_finder._data[f]) - 1 for f in corresp_finder._data])

    for family, dataset in corresp_finder.evals:
        metadata_dict["eval_{}_{}".format(family, dataset)] = corresp_finder.evals[(family, dataset)]

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in corresp_finder.errors:
            errorfile.write(",".join(line) + "\n")