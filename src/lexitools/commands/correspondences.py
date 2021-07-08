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
from collections import Counter, defaultdict
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

LEXICORE = [
    # ('lexibank', 'aaleykusunda'),
    # ('lexibank', 'abrahammonpa'),
    ('lexibank', 'allenbai'),
    ('lexibank', 'sagartst'),
    ('lexibank', 'starostinkaren'),
    # ('lexibank', 'bdpa'),
    # ('lexibank', 'beidasinitic'),
    # ('lexibank', 'birchallchapacuran'),
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
    # ('lexibank', 'lieberherrkhobwa'), ('lexibank', 'lundgrenomagoa'),
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
    # ('lexibank', 'walworthpolynesian'),
    # ('lexibank', 'wold'), ('lexibank', 'yanglalo'),
    # ('lexibank', 'zgraggenmadang'), ('lexibank', 'zhaobai'),
    # ('sequencecomparison', 'zhivlovobugrian'),
    # ('lexibank', 'backstromnorthernpakistan'),
    # ('lexibank', 'baf2'),
    # ('lexibank', 'clarkkimmun'),
    # ('lexibank', 'housinitic'),
    # ('lexibank', 'hsiuhmongmien'),
    # ('lexibank', 'lamayi'),
    # ('lexibank', 'liusinitic'),
    # ('lexibank', 'mortensentangkhulic'),
    # ('lexibank', 'polyglottaafricana'),
    # ('lexibank', 'simsrma'),
    # ('lexibank', 'starostinhmongmien'),
    # ('lexibank', 'tppsr'),
    # ('lexibank', 'vanbikkukichin'),
    # ('lexibank', 'wangbai'),
    # ('lexibank', 'wheelerutoaztecan'),
    # ('lexibank', 'wichmannmixezoquean'),
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


class FlyWeight(type):
    """ This is a Flyweight metaclass.

        This is used to cache objects, so that a single instance exists for a given set of
        keyword arguments (in a dataclass). The Flyweight pattern is very memory efficient.
    """

    def __init__(self, *args):
        super().__init__(args)
        self.cache = {}

    def __call__(self, **kwargs):
        id_ = tuple(sorted(kwargs.items()))
        try:
            return self.cache[id_]
        except KeyError:
            self.cache[id_] = super().__call__(**kwargs)
            return self.cache[id_]


@dataclass(eq=True, frozen=True)
class Lang(metaclass=FlyWeight):
    """ A Language has a name, a glottocode, and a family.

        This class is FlyWeight: we don't want duplicate objects for the exact
        same language.
    """
    __slots__ = ('glottocode', 'family', 'name')
    glottocode: str
    family: str
    name: str


@dataclass(eq=True, frozen=True)
class Sound(metaclass=FlyWeight):
    """ Sound in a correspondence, in a specific  languge, and context.

        This class is FlyWeight: we don't want duplicate objects for the exact
        same sound in context.

    Attributes:
        sound (str) : the sound which was observed in a correspondence
        lang (Lang) : the language in which this sound was observed
        context (str) : the context in which this sound was observed
    """
    __slots__ = ('sound', 'lang', 'left_context', 'right_context')
    sound: str
    lang: Lang
    left_context: str
    right_context: str


def wordlist_subset(lex, f):
    out = {}
    for i, row in lex._data.items():
        if i == 0:
            out[i] = row
        if f(i, row):
            out[i] = row
    return out


class CognateFinder(object):
    """ Loads lexibank data, organized by families, to facilitate the counting of correspondences.

    Attributes:
        errors (list of list): table which summarizes errors encountered in tokens.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of Lang to concepts.
        _data (defaultdict): mapping of (lang, concept) -> token string -> word instances
           A language can have more that one token for a specific
         concept, and a single token can be found in more than one dataset. A Word represents
         exactly one token in a given language and dataset.
    """

    def __init__(self, db, coarsen, glottolog, concept_list=None):
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

        # dict of family -> lingpy internal Wordlist dict
        self._data = defaultdict(lambda: {0: namespace})

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

        for idx, row in progressbar(enumerate(db.iter_table('FormTable')),
                                    desc="Loading data..."):

            idx = idx + 1  # 0 is the header row, so we need to offset all data indexes

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
            cogid = row.get("Cognacy", None)
            if cogid is None:
                cogid = row.get("Cognateset_ID", None)

            # Build internal dataset per family
            self._data[language.family][idx] = [
                row["Language_ID"],  # doculect
                concept_gloss,
                concept_id,
                " ".join(row["Segments"]),
                row["ID"],
                tokens,
                language.glottocode,
                # extract glottocode for lingpy ?
                cogid,
                row["dataset"]]

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

    def __getitem__(self, args):
        cols = list(self.cols)
        family, item = args
        if type(item) is int:
            return dict(zip(cols, self._data[family][item]))
        return [self[family, i] for i in item]

    def sanity_stats(self, lex):
        def mut_cov():
            for i in range(lex.height, 1, -1):
                if mutual_coverage_check(lex, i):
                    return i
            return 0

        d = {"min_mutual_coverage": mut_cov(),
             "average_coverage": average_coverage(lex)}
        return d

    def find_cognates(self, eval=False):
        lingpy.log.get_logger().setLevel(logging.WARNING)
        pbar = progressbar(self._data, desc="looking for cognates...")
        eval_measures = ["precision", "recall", "f-score"]

        for family in pbar:

            # Define the current list of cognate ids (if any)
            # Decide whether we need partial or wholistic cognate detection
            cogids = Counter()
            segmented = False
            for i, row in self._data[family].items():
                cogid = row[self.cols["cogid"]]
                if "+" in row[self.cols["tokens"]]:
                    segmented = True
                cogids[cogid] += 1

            if segmented:  # Cognates and alignments are morpheme-level
                lex = partial.Partial(self._data[family], check=True)
                pbar.set_description(
                    "looking for partial cognates in family %s (%s rows without gold cogids)" % (
                    family, cogids[None]))
                self.stats[family] = {"detection-type": 'partial'}

            else:  # Cognates and alignments are word-level
                lex = lingpy.LexStat(self._data[family], check=True)
                self.stats[family] = {"detection-type": 'LexStat'}

            self.stats[family].update(self.sanity_stats(lex))

            # prediction needed when we don't have all cogids, or when evaluating
            # if None in cogids or eval:
            # Decide which algorithm to run
            kw = dict(method='lexstat', threshold=0.55, ref="pred_cogid",
                      cluster_method='infomap')

            if self.stats[family]["average_coverage"] < .80 \
                    or self.stats[family]["min_mutual_coverage"] < 100:
                kw = dict(method="sca", threshold=0.45, ref='pred_cogid')
            self.stats[family]["lexstat_params"] = " ".join(
                "=".join([k, str(v)]) for k, v in kw.items())

            # Run cognate detection

            lex.get_scorer(runs=100)

            if segmented:
                lex.partial_cluster(**kw)
            else:
                lex.cluster(**kw)
            # else:
            #     # no prediction, just copy gold
            #     align_kw = dict(method = 'progressive')
            #     lex.add_entries("pred_cogid", "cogid", lambda x: [int(i) for i in x.split("-")])


            ## Evaluate cognate detection
            if eval:
                # separate data by dataset, keep only if gold annotation exists
                columns = lex.columns
                by_datasets = defaultdict(lambda: [list(columns)])
                for i, r in self._data[family].items():
                    if i > 0 and r[self.cols['cogid']] is not None:
                        dataset = r[self.cols['cldf_dataset']]
                        by_datasets[dataset].append(r)

                # Evaluate inside each dataset
                for dataset in by_datasets:
                    pbar.set_description("evaluating against gold rows in %s" % dataset)
                    gold_rows = dict(enumerate(by_datasets[dataset]))
                    eval_lex = lingpy.LexStat(gold_rows)
                    this_segmented = any("-" in cogid for i, cogid in eval_lex.iter_rows("cogid"))

                    if segmented and this_segmented:
                        res = lingpy.evaluate.acd.partial_bcubes(eval_lex, gold='cogid',
                                                                 test='pred_cogid',
                                                                 pprint=False)
                    else:
                        res = lingpy.evaluate.acd.bcubes(eval_lex, gold='cogid',
                                                         test='pred_cogid',
                                                         pprint=False)
                    d = dict(zip(eval_measures, res))
                    d.update(self.sanity_stats(eval_lex))
                    d["lexstat_params"] = self.stats[family]["lexstat_params"]
                    self.evals[(family, dataset)] = d

            ## Align and yield pairs of rows and alignments
            if segmented:
                # We don't want fuzzy alignments, we want independent alignments of morphemes
                # Because we want gaps to mean "deletion"/"insertion"
                # The workaround is to have rows for morphemes instead of tokens when segmentable
                # Note: this will result in a lot of duplication, so we keep a single
                # entry for each combination of morpheme, predicted cogid, and language.

                # (language, concept, morpheme, id): row
                rows = {}
                tok = self.cols["tokens"]
                lg = self.cols["doculect"]
                cog = len(self.cols)
                columns = list(self.cols)+["pred_cogid"]
                for i, *r in lex.iter_rows(*columns):
                    tokens = r[tok]
                    lang = r[lg]
                    pred_ids =  r[cog]

                    if type(pred_ids) is list and "+" in tokens:
                        morphemes = [list(s) for is_plus,s in groupby(tokens,lambda x:x=='+') if not is_plus]
                        if len(morphemes) == len(pred_ids):
                            # make a separate row for each morpheme
                            for morph, pred_subid in zip(morphemes, pred_ids):
                                new_row = list(r)
                                new_row[tok] = morph
                                new_row[cog] = pred_subid
                                rows[(lang, " ".join(morph), pred_subid)] = new_row
                        else:
                            r[cog] = pred_ids[0]
                            rows[(lang, " ".join(tokens), pred_ids[0])] = r
                    else:
                        r[cog] = pred_ids[0]
                        rows[(lang, " ".join(tokens), pred_ids[0])] = r

                lex = lingpy.LexStat(dict(enumerate([columns] + list(rows.values())))) # has less info and no scorer.

            alm = lingpy.Alignments(lex, ref="pred_cogid")
            align_kw = dict(method = 'progressive')
            alm.align(**align_kw)

            columns = list(alm.columns)
            for cognate_idx in alm.etd["pred_cogid"]:
                # list concatenation of the cognate indexes
                row_ids = sum(
                    [x for x in alm.etd["pred_cogid"][cognate_idx] if x != 0], [])
                rows = [dict(zip(columns, alm[i])) for i in row_ids]
                alignments = [alm[i, "alignment"] for i in row_ids]


                yield rows, alignments


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
                if (i == 0 or i ==l-1) and segment == "+":  # ignore initial and final "+"
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
                    if cat in {"tone"}:
                        yield None
                    else:
                        yield cat
            yield "#"

        def get_context(cats, i, sound, left):
            """Return the context for a given sound."""
            if cats[i] is None:  # No context for null "-" and tones
                return '', ''
            else:
                right = next((c for c in cats[i + 1:] if c is not None))
                return left, right

        cats = list(to_categories(seq))
        l = len(seq)
        prev = "#"
        for i in range(l):
            s = seq[i]
            l, r = get_context(cats, i, s, prev)
            prev = prev if cats[i] is None else cats[i]

            yield (s, (l, r))

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
        for words, aligned in self.data.find_cognates(eval=self.eval_cognates):

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
                event = tuple(sorted(sounds, key=lambda s: s.lang.glottocode))
                self.counts[event] += 1

                event_in_dataset = frozenset({(sounds[i], words[i]["cldf_dataset"])
                                              for i in range(l)})

                if len(self.examples[event]) < 5 and \
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
    db = MockLexicore(dataset_list)

    data = CognateFinder(db, coarse, pyglottolog.Glottolog(args.glottolog.dir),
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
            ["family", "min_mutual_coverage", "average_coverage", "lexstat_params"])
        for family in corresp_finder.data.stats:
            infos = corresp_finder.data.stats[family]
            writer.writerow(
                [family, infos["min_mutual_coverage"], infos["average_coverage"],
                 infos["lexstat_params"]])

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
