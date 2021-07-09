"""
Extract correspondences from a set of datasets.
"""
from dataclasses import dataclass, asdict

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
from itertools import groupby

# Do not merge consecutive vowels into diphthongs
lingpy.settings.rcParams["merge_vowels"] = False

# import cProfile
from memory_profiler import profile

LEXICORE = [

    ('lexibank', 'allenbai'),
    ('lexibank', 'sagartst'),
    ('lexibank', 'starostinkaren'),
    ('lexibank', 'bdpa'),
    ('lexibank', 'birchallchapacuran'),
    ('lexibank', 'lieberherrkhobwa'),
    ('lexibank', 'walworthpolynesian'),
    ('lexibank', 'yanglalo'),
    ('lexibank', 'baf2'),
    ('lexibank', 'wichmannmixezoquean'),
    ('lexibank', 'starostinhmongmien'),

    # ('lexibank', 'aaleykusunda'),
    # ('lexibank', 'abrahammonpa'),
    # ('lexibank', 'beidasinitic'),
    # ('sequencecomparison', 'blustaustronesian'),
    # ('lexibank', 'bodtkhobwa'), ('lexibank', 'bowernpny'),
    # ('lexibank', 'cals'), ('lexibank', 'castrosui'),
    # ('lexibank', 'castroyi'), ('lexibank', 'chaconarawakan'),
    # ('lexibank', 'chaconbaniwa'), ('lexibank', 'chaconcolumbian'),
    # ('lexibank', 'chenhmongmien'), ('lexibank', 'chindialectsurvey'),
    # ('lexibank', 'davletshinaztecan'), ('lexibank', 'deepadungpalaung'),
    # ('lexibank', 'dravlex'), ('lexibank', 'dunnaslian'),
    # ('sequencecomparison', 'dunnielex'), ('lexibank', 'galuciotupi'),
    # ('lexibank', 'gerarditupi'), ('lexibank', 'halenepal'),
    # ('lexibank', 'hantganbangime'),
    # ('sequencecomparison', 'hattorijaponic'),
    # ('sequencecomparison', 'houchinese'),
    # ('lexibank', 'hubercolumbian'), ('lexibank', 'ivanisuansu'),
    # ('lexibank', 'johanssonsoundsymbolic'),
    # ('lexibank', 'joophonosemantic'),
    # ('sequencecomparison', 'kesslersignificance'),
    # ('lexibank', 'kraftchadic'), ('lexibank', 'leekoreanic'),
    # ('lexibank', 'lundgrenomagoa'),
    # ('lexibank', 'mannburmish'), ('lexibank', 'marrisonnaga'),
    # ('lexibank', 'mcelhanonhuon'), ('lexibank', 'mitterhoferbena'),
    # ('lexibank', 'naganorgyalrongic'), ('lexibank', 'northeuralex'),
    # ('lexibank', 'peirosaustroasiatic'),
    # ('lexibank', 'pharaocoracholaztecan'), ('lexibank', 'robinsonap'),
    # ('lexibank', 'savelyevturkic'),
    # ('lexibank', 'sohartmannchin'),
    # ('sequencecomparison', 'starostinpie'), ('lexibank', 'suntb'),
    # ('lexibank', 'transnewguineaorg'), ('lexibank', 'tryonsolomon'),
    # ('lexibank', 'walkerarawakan'),
    # ('lexibank', 'wold'),
    # ('lexibank', 'zgraggenmadang'), ('lexibank', 'zhaobai'),
    # ('sequencecomparison', 'zhivlovobugrian'),
    # ('lexibank', 'backstromnorthernpakistan'),
    # ('lexibank', 'clarkkimmun'),
    # ('lexibank', 'housinitic'),
    # ('lexibank', 'hsiuhmongmien'),
    # ('lexibank', 'lamayi'),
    # ('lexibank', 'liusinitic'),
    # ('lexibank', 'mortensentangkhulic'),
    # ('lexibank', 'polyglottaafricana'),
    # ('lexibank', 'simsrma'),
    # ('lexibank', 'tppsr'),
    # ('lexibank', 'vanbikkukichin'),
    # ('lexibank', 'wangbai'),
    # ('lexibank', 'wheelerutoaztecan'),
    # ('sequencecomparison', 'listsamplesize')
]

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

""" Sound in a correspondence, in a specific  language, and context.

    This class is FlyWeight: we don't want duplicate objects for the exact
    same sound in context.

Attributes:
    sound (str) : the sound which was observed in a correspondence
    lang (Lang) : the language in which this sound was observed
    right_context (str) : the category of the closest non-tone segment on the right
    left_context (str) : the category of the closest non-tone segment on the left
"""
Sound = namedtuple("Sound", ('sound', 'lang', 'left_context', 'right_context'))


class MockLexicore(object):
    """ Mock interface to lexicore datasets.

    This loads a list of datasets. If not installed, it tries to pip install them,
    then the script needs to be re-ran. This is a temporary fix in order to use Lexicore
    before it really exists. The interface was agreed upon so that we can plug  in the
    correct interface soon with little effort.

    Attributes:
        datasets (dict of str to cldfbench.Dataset): maps dataset names to Dataset objects.
    """

    def __init__(self, dataset_list):
        """ Load all datasets from a list of dataset indentifiers, or install them.

        Note: this does NOT update installed datasets which are out of date.

        Args:
            dataset_list (list): list of dataset identifiers. Each identifier is a 2-tuple
                of a github organization name ("lexibank" or "sequencecomparison" in most
                cases) and a lexibank git repository name (such as "cals").
        """
        # see https://github.com/chrzyki/snippets/blob/main/lexibank/install_datasets.py
        egg_pattern = "git+https://github.com/{org}/{name}.git#egg=lexibank_{name}"
        successful_install = []
        failed_install = []
        for org, name in dataset_list:
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

        self.datasets = {}
        for org, name in dataset_list:
            try:
                self.datasets[name] = get_dataset(name,
                                                  ep="lexibank.dataset").cldf_reader()
            except AttributeError:
                print("Failed to load", name)
        if not self.datasets:
            raise Exception("No dataset loaded")

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


class CognateFinder(object):
    """ Prepare lexicore data, search and align cognates.

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
                     'original_id', 'tokens', 'glottocode',
                     'cogid', 'cldf_dataset']
        self.cols = {n: i for i, n in enumerate(namespace)}
        self.evals = {}
        self.stats = {}
        self.prefix = prefix

        # dict of family -> lingpy internal Wordlist dict
        self._data = defaultdict(lambda: [namespace])

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

        # We need to convert all cogids to ints, and often they're unfortunately strings
        cog_to_int = {}
        next_cogid = 1

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
            except ValueError:
                continue  # unknown sounds
            if all([s in IGNORE for s in tokens]): continue

            language = self.languages[row["Language_ID"]]
            dataset = row["dataset"]

            cogid = row.get("Cognacy", row.get("Cognateset_ID", None))
            if cogid in {"0", 0}:
                # it looks like 0 is a null value, si sagartst !
                cogid = None
            elif type(cogid) is str and " " in cogid:
                cogids = []
                for i in cogid.split(" "):
                    if (dataset, i) in cog_to_int:
                        cogids.append(cog_to_int[(dataset, i)])
                    else:
                        cog_to_int[(dataset, i)] = i
                        cogids.append(cog_to_int[(dataset, i)])
                        next_cogid += 1
                cogid = cogids
            elif (dataset, cogid) in cog_to_int:
                cogid = cog_to_int[(dataset, cogid)]
            else:
                cog_to_int[(dataset, cogid)] = next_cogid
                cogid = cog_to_int[(dataset, cogid)]
                next_cogid += 1

            # Build internal dataset per family
            self._data[language.family].append([
                row["Language_ID"],  # doculect
                concept_gloss,
                concept_id,
                " ".join(row["Segments"]),
                row["ID"],
                tokens,
                language.glottocode,
                cogid,
                dataset])

        # lingpy wordlist internal format
        self._data = {f:dict(enumerate(self._data[f])) for f in self._data}

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
            languages[row["ID"]] = Lang(family=family, glottocode=gcode, name=row[
                "Name"])  # TODO: should we use the langoid name here ?
        return languages

    def sanity_stats(self, lex):
        def mut_cov():
            for i in range(lex.height, 1, -1):
                if mutual_coverage_check(lex, i):
                    return i
            return 0

        d = {"min_mutual_coverage": mut_cov(),
             "average_coverage": average_coverage(lex)}
        return d

    def _find_cognates(self, family, pbar, eval, segmented, needs_annotation,
                       gold_segmented):
        # Define the current list of cognate ids (if any)
        # Decide whether we need partial or wholistic cognate detection


        if segmented:  # Cognates and alignments are morpheme-level
            lex = partial.Partial(self._data[family], check=True)
        else:  # Cognates and alignments are word-level
            lex = lingpy.LexStat(self._data[family], check=True)

        if lex.height == 0:
            datasets = {r[self.cols["cldf_dataset"]] for r in self._data[family]}
            print("Family %s has no usable data (datasets: {})".format(", ".join(datasets)))
            return None

        # Clear some memory...
        del self._data[family]

        self.stats[family] = self.sanity_stats(lex)

        # prediction needed when we don't have all cogids, or when evaluating
        if needs_annotation or eval:
            # Decide which algorithm to run
            kw = dict(method='lexstat', threshold=0.55, ref="pred_cogid",
                      cluster_method='infomap')

            if self.stats[family]["min_mutual_coverage"] < 100:
                kw = dict(method="sca", threshold=0.45, ref='pred_cogid',
                          no_bscorer=True)

            # Run cognate detection
            if segmented:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.stats[family]["detection_type"]  = 'morpheme',
                self.stats[family]["cognate_source"] = "Partial(" + kw_str + ")"
                pbar.set_description("looking for partial cognates in family %s" % family)
                lex.get_scorer(runs=1000)
                lex.partial_cluster(**kw)
            else:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.stats[family]["detection_type"]  = 'word',
                self.stats[family]["cognate_source"] = "LexStat(" + kw_str + ")"
                pbar.set_description("looking for cognates in family %s" % family)
                lex.get_scorer(runs=1000)
                lex.cluster(**kw)

        ## Evaluate cognate detection
        if eval:
            pbar.set_description("evaluating in %s" % family)
            self._eval_per_dataset(lex, family, gold_segmented, segmented)

        # Either we didn't predict, or we predicted only for evaluation purposes
        if not needs_annotation:
            self.stats[family]["detection_type"] = 'morpheme' if gold_segmented else 'word'
            lex.add_entries("pred_cogid", "cogid", lambda x: x, override=True)
            self.stats[family]["cognate_source"] = "expert"

        return lex

    def _eval_per_dataset(self, lex, family, gold_segmented, segmented):
        eval_measures = ["precision", "recall", "f-score"]
        # separate data by dataset, keep only if gold annotation exists
        columns = lex.columns
        by_datasets = defaultdict(lambda: [list(columns)])
        for i, r in lex._data.items():
            if i > 0 and r[lex._header['cogid']] is not None:
                dataset = r[lex._header['cldf_dataset']]
                by_datasets[dataset].append(list(r)) # copy of rows, otherwise we edit lex

        # Evaluate inside each dataset
        for dataset in progressbar(list(by_datasets),
                                   desc="dataset %s" % dataset):
            gold_rows = dict(enumerate(by_datasets[dataset]))

            eval_lex = lingpy.Wordlist(gold_rows)

            if segmented and gold_segmented:
                # TODO: check that the format is indeed the expected format for partial cogids
                res = lingpy.evaluate.acd.partial_bcubes(eval_lex, gold='cogid',
                                                         test='pred_cogid',
                                                         pprint=False)
            else:
                if segmented:  # revert to word-level cognates
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
            d["detection_type"] = "morpheme" if segmented else "word"
            self.evals[(family, dataset)] = d
            del by_datasets[dataset]

    def _iter_aligned_cognatesets(self, eval=False):
        lingpy.log.get_logger().setLevel(logging.WARNING)
        pbar = progressbar(list(self._data),
                           desc="looking for cognates...")

        for family in pbar:
            cogids_counts = Counter()
            segmented = False
            gold_segmented = False
            for i, row in self._data[family].items():
                cogid = row[self.cols["cogid"]]
                if "+" in row[self.cols["tokens"]]:
                    segmented = True
                if type(cogid) is list:
                    gold_segmented = True
                cogids_counts[cogid] += 1
            needs_annotation = None in cogids_counts

            # Align cognates
            partial = (needs_annotation and segmented) or (gold_segmented)
            lex = self._find_cognates(family, pbar, eval, segmented,
                                                        needs_annotation, gold_segmented)
            if lex is None: continue
            alm = lingpy.Alignments(lex,
                                    ref="pred_cogid", fuzzy=partial)
            pbar.set_description("Aligning cognates in %s" % family)
            align_kw = dict(method='library', iteration=True,
                            model="sca", mode="global")
            alm.align(**align_kw)

            cols = ["tokens", "original_token", "cldf_dataset",
                    "original_id", "glottocode", "doculect"]

            # Prepare aligned cognates
            pbar.set_description("iterating over alignments in %s" % family)

            for i, aligned in alm.get_msa("pred_cogid").items():
                alignments = aligned['alignment']
                rows = [dict(zip(cols, (alm[i, c] for c in cols))) for i in aligned["ID"]]


                # docs_l = max([len(r["doculect"]) for r in rows])
                # print("\n\n-------------------------------------------------------------"
                #       "-----------------------------------------------------------------")
                # print(alm[aligned["ID"][0], "concept"])
                # print("\n")
                # for i in range(len(alignments)):
                #     print(rows[i]["doculect"].rjust(docs_l," ")+"  ",*alignments[i], sep="\t")
                # input()

                yield rows, alignments

            del alm

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
        '--dataset',
        action='store',
        default='lexicore',
        help='select a specific lexibank dataset (otherwise entire lexicore)')
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


class Correspondences(object):
    """Extract sound correspondences.

    Attributes:
        args: the full args passed to the correspondences command.
        data (CognateFinder): the lexicore dataset
        clts (pyclts.CLTS): a clts instance
        bipa_cache (dict): maps strings to bipa sounds.
        counts (Counter): occurences of pairs of Sounds (the keys are frozensets).
        examples (defaultdict): example source words for pairs of sounds (the keys are frozensets).
        tones (set): characters which denote tones. Allow for a fast identification.
            (using calls to bipa is too slow)
        get_features (func): function to get sound features
        eval_cognates (bool): whether to evaluate cognacy detection.
    """

    def __init__(self, args, data, eval_cognates=False):
        """ Initialization only records the arguments and defines the attributes.

        Args:
            args: the full args passed to the correspondences command.
            data (CognateFinder): the data
            eval_cognates (bool): whether to evaluate cognacy detection.
            cognate_confusion_matrix (list of list): confusion matrix
            features_func (func): function to get sound features
        """
        self.args = args
        self.data = data
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.tones = set("⁰¹²³⁴⁵˥˦˧˨↓↑↗↘")
        self.cognates_pairs_by_datasets = Counter()
        self.score_cache = {}
        self.eval_cognates = eval_cognates

    def add_contexts(self, seq):
        """ Iterator of sounds and contexts for a pair of aligned tokens.

        Args:
            seq (list of str): sequence of segments or gap.

        Yields: pair of aligned sounds and their contexts: `(sound, context)`
        """

        def to_categories(sequence):
            """Turn a sequence of sounds into a sequence of categories used in contexts"""
            for s in sequence:
                if s == "-" or s in IGNORE:
                    yield None
                else:
                    cat = self.data.coarsen.category(s)
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
            left = left if cats[i] is None else cats[i]
            yield (seq[i], (left, right))

    def find_attested_corresps(self):
        """ Find all correspondences attested in our data.

            We record a correspondence for each aligned position, unless a sound is to be ignored.

            This functions returns None, but changes `self.corresps` in place.
        """

        def format_ex(r):
            template = "{word} ([{original_token}] {cldf_dataset} {original_id})"
            return template.format(word=" ".join(r["tokens"]), **r)

        self.args.log.info('Counting attested corresp...')
        exs_counts = Counter()

        # self.counts[
        for words, aligned in self.data._iter_aligned_cognatesets(
                eval=self.eval_cognates):

            l = len(words)
            if l <= 1: continue

            columns = list(zip(*[self.add_contexts(seq) for seq in aligned]))
            langs = [self.data.languages[w["doculect"]] for w in words]

            for sounds_and_contexts in columns:
                sounds, contexts = zip(*sounds_and_contexts)

                # Only markers
                if set(sounds) in ({"-"}, {"+"}): continue

                sounds = [Sound(lang=langs[i], sound=sounds[i],
                                left_context=contexts[i][0], right_context=contexts[i][1])
                          for i in range(l)]
                event = tuple(sorted(sounds))
                self.counts[event] += 1

                event_in_dataset = frozenset({(sounds[i], words[i]["cldf_dataset"])
                                              for i in range(l)})

                if len(self.examples[event]) < 3 and \
                        exs_counts[event_in_dataset] < 2:
                    exs_counts[event_in_dataset] += 1
                    examples = [format_ex(w) for w in
                                sorted(words, key=lambda w: w["glottocode"])]
                    self.examples[event].append(examples)


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

    coarse = Coarsen(clts.bipa, "src/lexitools/commands/default_coarsening.csv")

    ## This is a temporary fake "lexicore" interface
    dataset_list = LEXICORE if args.dataset == "lexicore" else [args.dataset.split("/")]

    data = CognateFinder(MockLexicore(dataset_list),
                         output_prefix, coarse,
                         pyglottolog.Glottolog(args.glottolog.dir),
                         concept_list=args.concepts)

    args.log.info(
        'Loaded the wordlist ({} languages, {} families, {} concepts kept)'.format(
            len(data.languages),
            len(data._data),
            len(data.concepts_subset)))

    corresp_finder = Correspondences(args, data, eval_cognates=args.cognate_eval)

    # with cProfile.Profile() as pr:
    corresp_finder.find_attested_corresps()

    # pr.dump_stats("profile.prof")

    with open(output_prefix + '_coarsening.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerows(coarse.as_table())

    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(
            ["family", "col_id", "count", "left_context", "sound", "right_context",
             "language", "glottocode", "examples"])
        for i, sounds in enumerate(corresp_finder.counts):
            col_id = sounds[0].lang.family + "-" + str(i)
            count = corresp_finder.counts[sounds]
            examples = corresp_finder.examples[sounds]

            for j, sound in enumerate(sounds):
                lang = sound.lang
                these_exs = "; ".join([ex[j] for ex in examples])
                writer.writerow([lang.family,
                                 col_id,
                                 count,
                                 sound.left_context,
                                 sound.sound,
                                 sound.right_context,
                                 lang.name,
                                 lang.glottocode,
                                 these_exs
                                 ])

    with open(output_prefix + '_cognate_info.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(
            ["family", "min_mutual_coverage", "average_coverage", "cognate_source",
             "cognate_type"])
        for family in corresp_finder.data.stats:
            infos = corresp_finder.data.stats[family]
            writer.writerow(
                [family, infos["min_mutual_coverage"], infos["average_coverage"],
                 infos["cognate_source"], infos["detection_type"]])

    metadata_dict = {"concepts": args.concepts,
                     "dataset": args.dataset}
    dataset_str = sorted(a + "/" + b for a, b in dataset_list)
    metadata_dict["dataset_list"] = dataset_str
    # metadata_dict["n_languages"] = len(data.lang_to_concept)
    metadata_dict["n_families"] = len(data._data)
    metadata_dict["n_concepts"] = len(data.concepts_subset)
    metadata_dict["n_tokens"] = sum([len(data._data[f]) - 1 for f in data._data])

    # TODO: update cognate eval
    for family, dataset in corresp_finder.data.evals:
        metadata_dict["eval_{}_{}".format(family, dataset)] = corresp_finder.data.evals[
            (family, dataset)]

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")
