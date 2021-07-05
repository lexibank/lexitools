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
from itertools import combinations
from collections import Counter, defaultdict
from pylexibank import progressbar
import pyglottolog
import csv
import csvw
import time
import subprocess
from importlib.util import find_spec
from cldfbench import get_dataset
import sys
import json
from lexitools.coarse_soundclass import Coarsen

import logging

# Do not merge consecutive vowels into diphthongs
lingpy.settings.rcParams["merge_vowels"] = False

# import cProfile

LEXICORE = [
    ('lexibank', 'aaleykusunda'),
    ('lexibank', 'abrahammonpa'),
    ('lexibank', 'allenbai'),
    ('lexibank', 'bdpa'),
    ('lexibank', 'beidasinitic'), ('lexibank', 'birchallchapacuran'),
    ('sequencecomparison', 'blustaustronesian'),
    ('lexibank', 'bodtkhobwa'), ('lexibank', 'bowernpny'),
    ('lexibank', 'cals'), ('lexibank', 'castrosui'),
    ('lexibank', 'castroyi'), ('lexibank', 'chaconarawakan'),
    ('lexibank', 'chaconbaniwa'), ('lexibank', 'chaconcolumbian'),
    ('lexibank', 'chenhmongmien'), ('lexibank', 'chindialectsurvey'),
    ('lexibank', 'davletshinaztecan'), ('lexibank', 'deepadungpalaung'),
    ('lexibank', 'dravlex'), ('lexibank', 'dunnaslian'),
    ('sequencecomparison', 'dunnielex'), ('lexibank', 'galuciotupi'),
    ('lexibank', 'gerarditupi'), ('lexibank', 'halenepal'),
    ('lexibank', 'hantganbangime'),
    ('sequencecomparison', 'hattorijaponic'),
    ('sequencecomparison', 'houchinese'),
    ('lexibank', 'hubercolumbian'), ('lexibank', 'ivanisuansu'),
    ('lexibank', 'johanssonsoundsymbolic'),
    ('lexibank', 'joophonosemantic'),
    ('sequencecomparison', 'kesslersignificance'),
    ('lexibank', 'kraftchadic'), ('lexibank', 'leekoreanic'),
    ('lexibank', 'lieberherrkhobwa'), ('lexibank', 'lundgrenomagoa'),
    ('lexibank', 'mannburmish'), ('lexibank', 'marrisonnaga'),
    ('lexibank', 'mcelhanonhuon'), ('lexibank', 'mitterhoferbena'),
    ('lexibank', 'naganorgyalrongic'), ('lexibank', 'northeuralex'),
    ('lexibank', 'peirosaustroasiatic'),
    ('lexibank', 'pharaocoracholaztecan'), ('lexibank', 'robinsonap'),
    ('lexibank', 'sagartst'), ('lexibank', 'savelyevturkic'),
    ('lexibank', 'sohartmannchin'),
    ('sequencecomparison', 'starostinpie'), ('lexibank', 'suntb'),
    ('lexibank', 'transnewguineaorg'), ('lexibank', 'tryonsolomon'),
    ('lexibank', 'walkerarawakan'),
    ('lexibank', 'walworthpolynesian'),
    ('lexibank', 'wold'), ('lexibank', 'yanglalo'),
    ('lexibank', 'zgraggenmadang'), ('lexibank', 'zhaobai'),
    ('sequencecomparison', 'zhivlovobugrian'),
    ('lexibank', 'backstromnorthernpakistan'),
    ('lexibank', 'baf2'),
    ('lexibank', 'clarkkimmun'),
    ('lexibank', 'housinitic'),
    ('lexibank', 'hsiuhmongmien'),
    ('lexibank', 'lamayi'),
    ('lexibank', 'liusinitic'),
    ('lexibank', 'mortensentangkhulic'),
    ('lexibank', 'polyglottaafricana'),
    ('lexibank', 'simsrma'),
    ('lexibank', 'starostinhmongmien'),
    ('lexibank', 'starostinkaren'),
    ('lexibank', 'tppsr'),
    ('lexibank', 'vanbikkukichin'),
    ('lexibank', 'wangbai'),
    ('lexibank', 'wheelerutoaztecan'),
    ('lexibank', 'wichmannmixezoquean'),
    ('sequencecomparison', 'listsamplesize')
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
    """ A Language has a name, a glottocode, a genus and a family.

        This class is FlyWeight: we don't want duplicate objects for the exact
        same language.
    """
    __slots__ = ('genus', 'glottocode', 'family', 'name')
    genus: str
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
    __slots__ = ('sound', 'lang', 'context')
    sound: str
    lang: Lang
    context: str


class SoundCorrespsByGenera(object):
    """ Loads lexibank data, organized by genera, to facilitate the counting of correspondences.

    Attributes:
        sound_class (func): a function which associates a class of sounds to any BIPA
            sound string.
        genera_to_lang (defaultdict): a mapping of each genus to Lang objects.
            Genera are loaded from Tiago Tresoldi's langgenera.
        errors (list of list): table which summarizes errors encountered in tokens.
        concepts_intersection (Counter): Counts the number of distinct concepts kept
            for each pair of languages encountered in a genera.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of Lang to concepts.
        _data (defaultdict): mapping of (lang, concept) -> token string -> word instances
           A language can have more that one token for a specific
         concept, and a single token can be found in more than one dataset. A Word represents
         exactly one token in a given language and dataset.
    """

    def __init__(self, db, langgenera_path, concept_list=None,
                 sound_class=None, glottolog=None):
        """ Initialize the data by collecting, processing and organizing tokens.

        Iterates over the database, to:
        - Identify genera and family name for all languages
        - If specified, restrict data to a specific list of concepts.
        - Pre-process tokens to replace sounds by sound classes and count syllables
        - Make note of any tokens which result in errors.

        Args:
            db (MockLexicore): Lexicore database
            langgenera_path (str): path to a tsv export of langgenera
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

        # dict of genus -> lingpy internal Wordlist dict
        self._data = defaultdict(lambda: {0: namespace})

        self.sound_class = sound_class

        with csvw.UnicodeDictReader(langgenera_path, delimiter="\t") as reader:
            lang_to_genera = {row['GLOTTOCODE']: row['GENUS'] for row in reader}

        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        self.languages = {}
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        self.errors = [["Dataset", "Language_ID", "Sound", "Token", "ID"]]
        self.concepts_intersection = Counter()
        self.populate_genera_to_lang(db, lang_to_genera, langoids)

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

            # Build internal dataset per genus
            self._data[language.genus][idx] = [
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

    def populate_genera_to_lang(self, db, lang_to_genera, langoids):
        self.genera_to_lang = defaultdict(set)
        for row in progressbar(db.iter_table("LanguageTable"), desc="listing languages"):
            gcode = row["Glottocode"]
            if gcode is None or gcode not in lang_to_genera:
                continue
            genus = lang_to_genera[gcode]
            family = row["Family"]
            langoid = langoids.get(gcode, None)
            if langoid is not None and langoid.family is not None:
                family = langoid.family.name

            lang = Lang(genus=genus, family=family, glottocode=gcode, name=langoid.name)
            self.genera_to_lang[genus].add(lang)
            self.languages[row["ID"]] = lang

    def __getitem__(self, args):
        cols = list(self.cols)
        genus, item = args
        if type(item) is int:
            return dict(zip(cols, self._data[genus][item]))
        return [self[genus, i] for i in item]

    def find_cognates(self, eval=False):
        lingpy.log.get_logger().setLevel(logging.WARNING)
        pbar = progressbar(self._data, desc="looking for cognates...")
        for genus in pbar:
            pbar.set_description("looking for cognates in genus %s" % genus)
            lex = lingpy.LexStat(self._data[genus], check=True)
            lex.get_scorer(runs=1000)
            lex.cluster(method='lexstat', threshold=0.55, ref="pred_cogid",
                        cluster_method='infomap')
            # lex.output('tsv', filename=file_template.format(genus))

            ## here create a new column based on pred and gold cogids ?

            alm = lingpy.Alignments(lex, ref="pred_cogid")
            alm.align(method='progressive', scoredict=lex.cscorer)

            for cognate_idx in alm.etd["pred_cogid"]:
                # list concatenation of the cognate indexes
                cognateset = sum(
                    [x for x in alm.etd["pred_cogid"][cognate_idx] if x != 0], [])
                rows = self[genus, cognateset]
                alignments = [alm[i, "alignment"] for i in cognateset]
                yield rows, alignments

            if eval:
                # separate data by dataset, keep only if gold annotation exists
                by_datasets = defaultdict(lambda: [self._data[genus][0]])

                for i, r in self._data[genus].items():
                    if i > 0 and r[self.cols['cogid']] is not None:
                        dataset = r[self.cols['cldf_dataset']]
                        by_datasets[dataset].append(r)

                # Evaluate inside each dataset
                # This re-runs lexstat, but it is the only clean way I found to evaluate
                for dataset in by_datasets:
                    pbar.set_description("evaluating against gold rows in %s" % dataset)
                    gold_rows = dict(enumerate(by_datasets[dataset]))
                    lex = lingpy.LexStat(gold_rows)
                    lex.get_scorer(runs=1000)
                    lex.cluster(method='lexstat', threshold=0.55, ref="pred_cogid",
                                cluster_method='infomap')

                    self.evals[dataset] = lingpy.evaluate.acd.bcubes(lex, gold='cogid',
                                                                     test='pred_cogid',
                                                                     pprint=False)

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
        for segment in tokens:
            try:
                if "/" in segment:
                    segment = segment.split("/")[1]
                yield self.sound_class(segment)
            except ValueError as e:
                self.errors.append((row["dataset"], row["Language_ID"], segment,
                                    " ".join(str(x) for x in segments), row["ID"]))
                raise e

    def __iter__(self):
        """Iterate over all (family, genus, lang, tokens) in the dataset.
        """
        for genus in self._data:
            for idx in self._data[genus]:
                if idx > 0:
                    row = self._data[genus][idx]
                    doculect = row[self.cols["doculect"]]
                    lang = self.languages[doculect]
                    yield lang.family, genus, doculect, row[self.cols["tokens"]]


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

    parser.add_argument(
        '--cutoff',
        action='store',
        default=0.05,
        type=float,
        help='Cutoff for attested correspondences in a language pair, in proportion of the list of cognates.')

    parser.add_argument(
        '--model',
        choices=["BIPA", "Coarse"],
        default='BIPA',
        type=str,
        help='select a sound class model: BIPA or Coarse.')
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
        data (SoundCorrespsByGenera): the lexicore dataset
        clts (pyclts.CLTS): a clts instance
        bipa_cache (dict): maps strings to bipa sounds.
        counts (Counter): occurences of pairs of Sounds (the keys are frozensets).
        examples (defaultdict): example source words for pairs of sounds (the keys are frozensets).
        total_cognates (Counter): counts the number of cognates found for each pair of languages.
        tones (set): characters which denote tones. Allow for a fast identification.
            (using calls to bipa is too slow)
        get_features (func): function to get sound features
        eval_cognates (bool): whether to evaluate cognacy detection.
    """

    def __init__(self, args, data, clts, features_func=None, eval_cognates=False):
        """ Initialization only records the arguments and defines the attributes.

        Args:
            args: the full args passed to the correspondences command.
            data (SoundCorrespsByGenera): the data
            clts (pyclts.CLTS): a clts instance
            eval_cognates (bool): whether to evaluate cognacy detection.
            cognate_confusion_matrix (list of list): confusion matrix
            features_func (func): function to get sound features
        """
        self.args = args
        self.data = data
        self.clts = clts
        self.sca_cache = {}
        self.bipa_cache = {}
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.total_cognates = Counter()
        self.tones = set("⁰¹²³⁴⁵˥˦˧˨↓↑↗↘")
        self.cognates_pairs_by_datasets = Counter()
        self.get_features = features_func
        self.score_cache = {}
        self.eval_cognates = eval_cognates

    def bipa(self, item):
        """ Caches calls to the bipa transcription system, as resolve_sound is too slow.

        Args:
            item: a string representing a sound

        Returns:
            bipa (Sound): the corresponding BIPA sound
        """
        try:
            return self.bipa_cache[item]
        except KeyError:
            self.bipa_cache[item] = self.clts.bipa[item]
            return self.bipa_cache[item]

    def find_available(self):
        """ Find which pairs of sounds from our data are available in each genera.

        - A pair of two distinct sounds x,y are available in a genus if the genus has at
         least two distinct languages A,B such that A has at least two occurences of x
         and B has at least two occurences of y.
        - A pair of a sound and a gap (x,-) is available in a genus if that genus has a
        language with at least two occurences of x.
        - A pair of a sound and itself (x,x) is available in a genus if that genus has a
         language with at least two occurences of x.

        Returns:
            available (list of lists): Inner lists are rows with [family, genus, soundA, soundB]
        """

        self.args.log.info('Counting available corresp...')

        sounds_by_genera = defaultdict(lambda: defaultdict(Counter))
        for family, genus, lang, tokens in self.data:
            for sound in tokens:
                if sound not in IGNORE:  # spaces and segmentation symbols ignored
                    sounds_by_genera[(family, genus)][sound][lang] += 1

        available = list()
        for family, genus in list(sounds_by_genera):
            freq = sounds_by_genera[(family, genus)]
            n_sounds = len(freq)
            tot_sound_pairs = (n_sounds * (n_sounds - 1)) / 2
            sound_pairs = combinations(freq, r=2)

            for sound_A in progressbar(freq):
                if sound_A != "-":  # No point in counting corresp between blank and itself
                    occ = {lg for lg in freq[sound_A] if freq[sound_A][lg] > 1}
                    if len(occ) > 1:
                        available.append([family, genus, sound_A, sound_A])
                    if len(occ) > 0:
                        available.append([family, genus, sound_A, "-"])

            for sound_A, sound_B in progressbar(sound_pairs, total=tot_sound_pairs):
                occ_A = {lg for lg in freq[sound_A] if freq[sound_A][lg] > 1}
                occ_B = {lg for lg in freq[sound_B] if freq[sound_B][lg] > 1}
                if occ_A and occ_B and len(occ_A | occ_B) > 1:
                    sound_A, sound_B = tuple(sorted((sound_A, sound_B)))
                    available.append([family, genus, sound_A, sound_B])

        return available

    def sounds_and_contexts(self, almA, almB):
        """ Iterator of sounds and contexts for a pair of aligned tokens.

        Args:
            almA (list of str): aligned elements of the first token.
            almB (list of str): aligned elements of the second token.

        Yields: pair of aligned sounds and their contexts: `(sA, cA), (sB, cB)`
            sA (str) and sB (str): aligned sounds from resp. almA and almB
            cA (str) and cB (str): contexts for resp. sA and sB
        """

        def to_categories(sequence):
            """Turn a sequence of sounds into a sequence of categories used in contexts"""
            for s in sequence:
                cat = self.bipa(s).type
                if s == "-" or s in IGNORE or cat in {"tone"}:
                    yield None
                elif cat in {"vowel", "diphthong"}:
                    yield "V"
                else:
                    yield "C"  # consonant or cluster
            yield "#"

        def get_context(cats, i, sound, left):
            """Return the context for a given sound."""
            if cats[i] is None:  # No context for null "-" and tones
                return sound
            else:
                right = next((c for c in cats[i + 1:] if c is not None))
                return left + sound + right

        catsA = list(to_categories(almA))
        catsB = list(to_categories(almB))
        l = len(almA)
        prevA, prevB = "#", "#"
        for i in range(l):
            sA = almA[i]
            cA = get_context(catsA, i, sA, prevA)
            prevA = prevA if catsA[i] is None else catsA[i]
            sB = almB[i]
            cB = get_context(catsB, i, sB, prevB)
            prevB = prevB if catsB[i] is None else catsB[i]
            yield (sA, cA), (sB, cB)

    #
    def find_attested_corresps(self):
        """ Find all correspondences attested in our data.

            We record a correspondence for each aligned position, unless a sound is to be ignored.

            This functions returns None, but changes `self.corresps` in place.
        """
        self.args.log.info('Counting attested corresp...')
        exs_counts = Counter()
        for words, aligned in self.data.find_cognates(eval=self.eval_cognates):
            l = len(words)

            # convert to pairwise
            for i, j in combinations(range(l), 2):
                wordA = words[i]
                wordB = words[j]

                # Increment total cognates per pair of languages
                langs = (self.data.languages[wordA["doculect"]],
                         self.data.languages[wordB["doculect"]])

                # ignore pairs across synonyms
                if langs[0] == langs[1]: continue

                self.total_cognates[langs] += 1

                # Record that we found a cognate pair for these datasets
                datasetA, datasetB = (wordA["cldf_dataset"], wordB["cldf_dataset"])
                self.cognates_pairs_by_datasets[tuple(sorted((datasetA, datasetB)))] += 1

                almA = aligned[i]
                almB = aligned[j]
                for (soundA, ctxtA), (soundB, ctxtB) in self.sounds_and_contexts(almA,
                                                                                 almB):
                    if not IGNORE.isdisjoint({soundA, soundB}):
                        continue

                    A = Sound(lang=langs[0], sound=soundA, context=ctxtA)
                    B = Sound(lang=langs[1], sound=soundB, context=ctxtB)
                    event = frozenset({A, B})
                    self.counts[event] += 1
                    event_in_dataset = frozenset({A, datasetA,
                                                  B, datasetB})
                    if len(self.examples[event]) < 5 and \
                            exs_counts[event_in_dataset] < 2:
                        exs_counts[event_in_dataset] += 1
                        self.examples[event].append((wordA, wordB))


def run(args):
    """Run the correspondence command.

    Run with:

        cldfbench lexitools.correspondences --clts-version v1.4.1 --model Coarse --cutoff 0.05 --dataset lexicore

    For details on the arguments, see `cldfbench lexitools.correspondences --help`.

    This loads all the requested datasets and searches for available sounds and attested
    correspondences. It output a series of files which start by a time-stamp,
    then "_sound_correspondences_" and end in:

    `_available.csv`: a csv table of available sounds. The header row is:
        `Family,Genus,Sound A,Sound B`. The order of Sound A and
        Sound B is not meaningful, we do not duplicate A/B and B/A.
    `_coarsening.csv`: output only if run with the model "Coarse". This is a table of all
        known coarse sounds.
    `_concepts_intersection.csv`: a csv table giving the number of common concepts
        in the data, and of concepts kept, for each pair of languages A and B.
        The header row is `Lang A,Lang B,Common concepts,Kept concepts`.
        The order of Lang A and Lang B is not meaningful, we do not duplicate A/B and B/A.
    `_counts.csv`: a csv table recording correspondences. The header row is:
    `Family,Genus,Lang A,Lang B,Sound A,Sound B,Env A,Env B,Count`.
        The order of languages A/B and sounds A/B is not meaningful,
        we do not duplicate A/B and B/A. Env A and B are the environments in which sounds
        were observed (their contexts).
    `_sound_errors.csv`: records sounds that raised errors in CLTS. The corresponding
        tokens have been ignored by the program. The header row is
        `Dataset,Language_ID,Sound,Token,ID`.
    `_metadata.json`: a json file recording the input parameters and all relevant metadata.
    """
    args.log.info(args)
    langgenera_path = "./src/lexitools/commands/lang_genera-v1.0.0.tsv"
    clts = args.clts.from_config().api
    now = time.strftime("%Y%m%d-%Hh%Mm%Ss")
    output_prefix = "{timestamp}_sound_correspondences".format(timestamp=now)

    """Three options for sound classes:
    1. Keep original BIPA symbols. Very precise, but variations in annotation make the results noisy.
    2. Coarsen BIPA to keep only some main features. This tries to strike a balance between BIPA and avoiding noise.
    """
    if args.model == "BIPA":
        def to_sound_class(sound):
            return str(clts.bipa[sound])

        def sound_features(sound):
            return clts.bipa[sound].featureset

    elif args.model == "Coarse":
        coarse = Coarsen(clts.bipa, "src/lexitools/commands/default_coarsening.csv")

        def to_sound_class(sound):
            return coarse[sound]

        def sound_features(sound):
            return coarse.features.get(sound, {})
    else:
        raise ValueError("Incorrect sound class model")

    ## This is a temporary fake "lexicore" interface
    dataset_list = LEXICORE if args.dataset == "lexicore" else [args.dataset.split("/")]
    db = MockLexicore(dataset_list)

    data = SoundCorrespsByGenera(db, langgenera_path,
                                 sound_class=to_sound_class,
                                 concept_list=args.concepts,
                                 glottolog=pyglottolog.Glottolog(
                                     args.glottolog.dir))

    args.log.info(
        'Loaded the wordlist ({} languages, {} genera, {} concepts kept)'.format(
            len(data.languages),
            len(data._data),
            len(data.concepts_subset)))

    corresp_finder = Correspondences(args, data, clts,
                                     features_func=sound_features,
                                     eval_cognates=args.cognate_eval)

    available = corresp_finder.find_available()

    # with cProfile.Profile() as pr:
    corresp_finder.find_attested_corresps()

    # pr.dump_stats("profile.prof")

    def format_ex(rows):
        r1, r2 = rows
        tok1 = " ".join(r1["tokens"])
        tok2 = " ".join(r2["tokens"])
        template = "{word} ([{original_token}] {cldf_dataset} {original_id})"

        return template.format(word=tok1, **r1) + "/" + \
               template.format(word=tok2, **r2)

    with open(output_prefix + '_coarsening.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerows(coarse.as_table())

    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["family", "genus", "lang_a", "lang_b", "sound_a",
                         "sound_b", "env_a", "env_b", "count", "examples"])
        for sounds in corresp_finder.counts:
            # ensure to always have the same order.
            A, B = sorted(sounds, key=lambda s: (s.sound, s.context))
            count = corresp_finder.counts[sounds]
            total = corresp_finder.total_cognates[(A.lang, B.lang)]
            if count > max(2, args.cutoff * total):
                examples = [format_ex(rows) for rows in corresp_finder.examples[sounds]]
                examples = "; ".join(examples)
                writer.writerow([A.lang.family, A.lang.genus,
                                 A.lang.glottocode, B.lang.glottocode,
                                 A.sound, B.sound, A.context, B.context, count, examples])

    pairs_in_dataset = 0
    pairs_across_datasets = 0
    with open(output_prefix + '_pairs_count_by_datasets.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["dataset_a", "dataset_b", "number_of_cognate_pairs"])
        for (dA, dB) in corresp_finder.cognates_pairs_by_datasets:
            count = corresp_finder.cognates_pairs_by_datasets[(dA, dB)]
            writer.writerow([dA, dB, count])
            if dA == dB:
                pairs_in_dataset += count
            else:
                pairs_across_datasets += count

    with open(output_prefix + '_available.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["family", "genus", "sound_a", "sound_B"])
        writer.writerows(available)

    with open(output_prefix + '_concepts_intersection.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["lang_a", "lang_b", "common_concepts", "kept_concepts"])
        for lA, lB in data.concepts_intersection:
            concepts = data.concepts_intersection[(lA, lB)]
            kept = corresp_finder.total_cognates[(lA, lB)]
            writer.writerow([lA.glottocode, lB.glottocode, concepts, kept])

    metadata_dict = {"observation cutoff": args.cutoff,
                     "model": args.model,
                     "concepts": args.concepts,
                     "dataset": args.dataset}
    dataset_str = sorted(a + "/" + b for a, b in dataset_list)
    metadata_dict["dataset_list"] = dataset_str
    # metadata_dict["n_languages"] = len(data.lang_to_concept)
    metadata_dict["n_genera"] = len(data.genera_to_lang)
    metadata_dict["n_concepts"] = len(data.concepts_subset)
    metadata_dict["n_tokens"] = len(data._data)
    metadata_dict["n_cognate_pairs_across_datasets"] = pairs_across_datasets
    metadata_dict["n_cognate_pairs_in_datasets"] = pairs_in_dataset
    metadata_dict["cutoff_method"] = "max(2, cutoff * shared_cognates)"

    # TODO: update cognate eval
    for dataset in corresp_finder.data.evals:
        p, r, f = corresp_finder.data.evals[dataset]
        metadata_dict["eval_{}_precision".format(dataset)] = p
        metadata_dict["eval_{}_recall".format(dataset)] = r
        metadata_dict["eval_{}_fscore".format(dataset)] = f

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")
