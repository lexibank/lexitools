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
from itertools import combinations, product
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
from typing import List

# Do not merge consecutive vowels into diphthongs
lingpy.settings.rcParams["merge_vowels"] = False

# import cProfile

LEXICORE = [('lexibank', 'aaleykusunda'), ('lexibank', 'abrahammonpa'),
            ('lexibank', 'allenbai'), ('lexibank', 'bdpa'),
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
            ('lexibank', 'walkerarawakan'), ('lexibank', 'walworthpolynesian'),
            ('lexibank', 'wold'), ('lexibank', 'yanglalo'),
            ('lexibank', 'zgraggenmadang'), ('lexibank', 'zhaobai'),
            ('sequencecomparison', 'zhivlovobugrian')
            ]

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


@dataclass
class Word:
    """ A word in a specific language and dataset.

    A word is linked to a dataset row and carries its ID, so that we can refer back to
    the data it originates from.

    Attributes:
        lang (Lang) : the language in which this token was observed
        token (str): the token
        concept (str): the concept denoted by the token (concepticon ID)
        syllables (int): number of syllables in the token
        original_token (str): the original lexibank token (from Segments)
        dataset (str): the dataset where this token was recorded
        id (str): the identifier for the row in the dataset's wordlist
    """
    __slots__ = ('lang', 'token', 'concept', 'syllables', 'original_token',
                 'dataset', 'id', 'cognate_id')
    lang: Lang
    token: List[str]
    concept: str
    syllables: int
    original_token: str
    dataset: str
    id: str
    cognate_id: str


class SoundCorrespsByGenera(object):
    """ Loads lexibank data, organized by genera, to facilitate the counting of correspondences.

    Attributes:
        sound_class (func): a function which associates a class of sounds to any BIPA
            sound string.
        genera_to_lang (defaultdict): a mapping of each genus to Lang objects.
            Genera are loaded from Tiago Tresoldi's langgenera.
        errors (list of list): table which summarizes errors encountered in tokens.
        asjp (bool): whether the only dataset is ASJP, in which case we will extract
            Graphemes rather than Segments.
        concepts_intersection (Counter): Counts the number of distinct concepts kept
            for each pair of languages encountered in a genera.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of Lang to concepts.
        data (defaultdict): mapping of (lang, concept) -> token string -> word instances
           A language can have more that one token for a specific
         concept, and a single token can be found in more than one dataset. A Word represents
         exactly one token in a given language and dataset.
    """

    def __init__(self, db, langgenera_path, concept_list=None,
                 sound_class=None, glottolog=None, asjp=False):
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
            asjp (bool): whether the only dataset is ASJP, in which case we will extract
            Graphemes rather than Segments.
        """
        self.sound_class = sound_class

        with csvw.UnicodeDictReader(langgenera_path, delimiter="\t") as reader:
            lang_to_genera = {row['GLOTTOCODE']: row['GENUS'] for row in reader}

        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        langs = {}
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        self.genera_to_lang = defaultdict(set)
        self.errors = [["Dataset", "Language_ID", "Sound", "Token", "ID"]]
        self.asjp = asjp
        self.concepts_intersection = Counter()

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
            langs[row["ID"]] = lang

        concepts = {row["ID"]: row["Concepticon_ID"] for row in
                    db.iter_table('ParameterTable')}

        if concept_list is not None:
            concepticon = Concepticon(Config.from_file().get_clone("concepticon"))
            concept_dict = concepticon.conceptlists[concept_list].concepts
            self.concepts_subset = {c.concepticon_id for c in
                                    concept_dict.values() if
                                    c.concepticon_id}
        else:  # All observed concepts
            self.concepts_subset = set(concepts.values())

        self.lang_to_concept = defaultdict(set)
        self.data = defaultdict(lambda: defaultdict(list))

        for row in progressbar(db.iter_table('FormTable'),
                               desc="Loading data..."):

            if row.get("Loan", ""): continue  # Ignore loan words

            concept = concepts[row["Parameter_ID"]]
            if concept not in self.concepts_subset or \
                    row["Language_ID"] not in langs or \
                    (not self.asjp and row["Segments"] is None) or \
                    (self.asjp and row["Graphemes"] is None):
                continue

            # TODO: if it has COGIDS, split on morphemes
            # TODO: add a Word for each morpheme + morpheme cogid

            try:
                token = list(self._iter_phonemes(row))
            except ValueError:
                continue  # unknown sounds
            if all([s in IGNORE for s in token]): continue

            syllables = len(lingpy.sequence.sound_classes.syllabify(token,
                                                                    output="nested"))

            lang = langs[row["Language_ID"]]

            # TODO: also add COGID
            word = Word(lang=lang, syllables=syllables,
                        token=token, concept=concept, id=row["ID"],
                        cognate_id=row.get("Cognateset_ID", None),
                        original_token=" ".join(row["Segments"]), dataset=row["dataset"])

            self.data[(lang, concept)][" ".join(token)].append(word)
            self.lang_to_concept[lang].add(concept)

    def _iter_phonemes(self, row):
        """ Iterate over pre-processed phonemes from a row's token.

        The phonemes are usually from the "Segments" column, except for ASJP data
        where we retrieve them from "Graphemes".

        We pre-process by:
            - re-tokenizing on spaces, ignoring empty segments
            - selecting the second element when there is a slash
            - using the sound_class attribute function to obtain sound classes

        Args:
            row (dict): dict of column name to value

        Yields:
            successive sound classes in the row's word.
        """
        # In some dataset, the separator defined in the metadata is " + ",
        # which means that tokens are not phonemes (ex:bodtkhobwa)
        # This is solved by re-tokenizing on the space...
        if self.asjp:
            segments = row["Graphemes"][1:-1]  # Ignore end and start symbols
        else:
            segments = row["Segments"]

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

    def iter_candidates(self):
        """ Iterate over word pair candidates.

        Across all datasets, inside each genus, we consider all token
        pairs for the same concept in all language pairs.

        Yields:
            tuples of `genus, (langA, tA, sA), (langB, tB, sB)`
                genus (str): genus name
                langA (str) and langB (str): glottocodes for the two languages
                tA (list of str) and tB (list of str): the two tokens
                sA (int) and sB (int): the syllable count for each token
        """
        for genus in progressbar(self.genera_to_lang, desc="Genera"):
            langs = self.genera_to_lang[genus]
            lang_pairs = combinations(langs, r=2)
            n_lang = len(langs)
            tot_pairs = (n_lang * (n_lang - 1)) / 2
            for langA, langB in progressbar(lang_pairs, total=tot_pairs,
                                            desc="Language pairs"):
                concepts_A = self.lang_to_concept[langA]
                concepts_B = self.lang_to_concept[langB]
                common_concepts = (concepts_A & concepts_B)
                self.concepts_intersection[(langA, langB)] += len(common_concepts)
                for concept in common_concepts:
                    for tokA, tokB in product(self.data[(langA, concept)],
                                              self.data[(langB, concept)]):
                        # Here we grab the first word, but there may be other words,
                        # if this token is documented in other datasets.
                        # So far we don't really need the information.
                        wordA = self.data[(langA, concept)][tokA][0]
                        wordB = self.data[(langB, concept)][tokB][0]
                        yield wordA, wordB

    def __iter__(self):
        """Iterate over the tokens.

        Yields:
            for all known tokens, its genus, language glottocode, concept, and the token itself.
        """
        for lang, concept in self.data:
            for token in self.data[(lang, concept)]:
                yield self.data[(lang, concept)][token][0]  # also picking a single word


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
        '--threshold',
        action='store',
        default=1,
        type=float,
        help='Max differences per syllable in the SCA string.')
    parser.add_argument(
        '--cutoff',
        action='store',
        default=0.05,
        type=float,
        help='Cutoff for attested correspondences in a language pair, in proportion of the list of cognates.')

    parser.add_argument(
        '--model',
        choices=["BIPA", "ASJPcode", "Coarse"],
        default='BIPA',
        type=str,
        help='select a sound class model: BIPA, ASJPcode, or Coarse.')

    parser.add_argument(
        '--alignment',
        choices=["sca", "simple"],
        default='simple',
        type=str,
        help='select an alignment method: either of SCA or a simple scorer '
             'which penalizes C/V matches and forbids T/C & T/V matches.')
    parser.add_argument(
        '--bdpa',
        action='store',
        default=None,
        type=str,
        help='path to BDPA gold alignments, for evaluation (psa file)')
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
        sca_cache (dict): maps bipa sounds to SCA class (used for the cognate threshold).
        bipa_cache (dict): maps strings to bipa sounds.
        counts (Counter): occurences of pairs of Sounds (the keys are frozensets).
        examples (defaultdict): example source words for pairs of sounds (the keys are frozensets).
        total_cognates (Counter): counts the number of cognates found for each pair of languages.
        tones (set): characters which denote tones. Allow for a fast identification.
            (using calls to bipa is too slow)
    """

    def __init__(self, args, data, clts, align_method):
        """ Initialization only records the arguments and defines the attributes.

        Args:
            args: the full args passed to the correspondences command.
            data (SoundCorrespsByGenera): the data
            clts (pyclts.CLTS): a clts instance
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
        if align_method == "sca":
            self.align = self.align_sca
        elif align_method == "simple":
            self.align = self.align_simple

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

    def sca(self, item):
        """ Caches calls to the SCA sound class system, as resolve_sound is too slow.
        Args:
            item: a string representing a sound

        Returns:
            sca (str): the corresponding SCA class
        """
        try:
            return self.sca_cache[item]
        except KeyError:
            self.sca_cache[item] = self.clts.soundclasses_dict["sca"][item]
            return self.sca_cache[item]

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
        for word in self.data:
            for sound in word.token:
                if sound not in IGNORE:  # spaces and segmentation symbols ignored
                    sounds_by_genera[(word.lang.family, word.lang.genus)][sound][
                        word.lang] += 1

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

    def allowed_differences(self, sa, sb):
        """ Compute the number of allowed differences for two syllable length.

        Args:
            sa (int): number of syllables in the first token
            sb (int): number of syllables in the second token

        Returns:
            diff (int): a threshold above which two words of these lengths
                can be considered cognates.
        """
        return max(sa, sb) * self.args.threshold

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

    def get_scorer(self, seqA, seqB):
        """ Returns an alignment scorer which penalizes tone alignments with non tones.

        The alignment scorer is a dict of a pair of sounds (a,b) to a score, for all pairs
        of sounds across the two sequences.

        Here, we set :

        - 1 for a match,
        - -10 for mismatches involving a tone and something that is not a tone,
        - -1.5 for other cross-category mismatches
        - -1 for in-category mismatches and indels

        The reason to penalize tones is that having tones in the sequence of sounds is
        only a notational trick, as they actually belong to a different tier.
        The reason to prefer in-category matches is to favor alignments of type CV/-V
        rather than CV/V-.

        For reference, the default lingpy scorer is:
        >>> {(a, b): 1.0 if a == b else -1.0 for a, b in product(seqA, seqB)}

        Args:
            seqA (iterable of str): first sequence
            seqB (iterable of str): second sequence

        Returns:
            scorer (dict): maps from pairs of sounds to score.
        """

        def score(a, b):
            if a == b:
                return 1
            a_cat = self.tones.isdisjoint(a)
            b_cat = self.tones.isdisjoint(b)
            if a_cat != b_cat:
                return -10
            elif self.bipa(a).type != self.bipa(b).type:
                return -1.5
            else:
                return -1

        return {(a, b): score(a, b) for a, b in product(seqA, seqB)}

    def are_cognate(self, wordA, wordB):
        """ Test if two words are cognates.

        If we have gold cognate identifier information, we simply use it.
        Otherwise, we assume tokens are cognates if they are similar enough.
        The similarity threshold is proportional to the number of syllables.
        The distance is an edit distance over SCA strings.
        This allows any changes insides SCA's classes for free,
        that is, historically expected changes are not penalized.

        TODO:
            - Add option to swap this out for other methods.
             Using a method which considers whole lists will require refactoring of
             SoundCorrespsByGenera.
            - Align partial cognates separately.

        Args:
            wordA (Word): the first word
            wordB (Word): the second word

        Returns:
            cognacy (bool): whether the two words should be considered cognates.
        """
        # In a single dataset with gold cognate IDs, rely on it
        if wordA.dataset == wordB.dataset and wordA.cognate_id is not None:
            return wordA.cognate_id == wordB.cognate_id

        # Identical words are cognate
        tokA, tokB = wordA.token, wordB.token
        if tokA == tokB: return True

        # Identical sequences of sound classes are cognates
        tokA = [self.sca(s) for s in tokA]
        tokB = [self.sca(s) for s in tokB]
        if tokA == tokB: return True

        # Use a threshold on sequence similarity
        allowed = self.allowed_differences(wordA.syllables, wordB.syllables)

        # Estimate a lower boundary by checking character sets.
        # If A has n more characters than B, and B m more,
        # and m > n, then we need at least m edits
        # (n substitutions, and m-n insertions)
        # Of course, it is likely to be more, but this is enough to
        # decide cases where tokens are very different
        sa, sb = set(tokA), set(tokB)
        lower_boundary = max(len(sa - sb), len(sb - sa))
        if lower_boundary > allowed: return False

        return lingpy.edit_dist(tokA, tokB) <= allowed  # This is the bottleneck

    def align_simple(self, a, b):
        """ Perform the alignment using a simple method and custom scorer."""
        return lingpy.nw_align(a, b, self.get_scorer(a, b))

    def align_sca(self, a, b):
        """ Perform the alignment using SCA from lingpy."""
        if a == b:
            return a, b, 0
        p = lingpy.Pairwise(a, b)
        p.align(model="sca")
        return p.alignments[0]

    def find_attested_corresps(self):
        """ Find all correspondences attested in our data.

        - Inside each genus, we consider all pairs of tokens for the same concept across two
        languages.
        - We apply a simple cognacy test and align cognate pairs using our custom scorer.
        - We record a correspondence for each aligned position, unless a sound is to be ignored.

        We do not use a better cognate recognition method or cognate alignment function,
        as these tend to insert too much knowledge, which we then find back identical in
        correspondences.

        This functions returns None, but changes `self.corresps` in place.
        """
        self.args.log.info('Counting attested corresp...')
        exs_counts = Counter()
        for wordA, wordB in self.data.iter_candidates():
            if self.are_cognate(wordA, wordB):
                self.total_cognates[(wordA.lang, wordB.lang)] += 1

                # Record that we found a cognate pair for these datasets
                datasets = tuple(sorted((wordA.dataset,wordB.dataset)))
                self.cognates_pairs_by_datasets[datasets] += 1

                almA, almB, sim = self.align(wordA.token, wordB.token)
                for (soundA, ctxtA), (soundB, ctxtB) in self.sounds_and_contexts(almA,
                                                                                 almB):
                    if IGNORE.isdisjoint({soundA, soundB}):
                        A = Sound(lang=wordA.lang, sound=soundA, context=ctxtA)
                        B = Sound(lang=wordB.lang, sound=soundB, context=ctxtB)
                        event = frozenset({A, B})
                        self.counts[event] += 1
                        event_in_dataset = frozenset({A, wordA.dataset,
                                                      B, wordB.dataset})
                        if len(self.examples[event]) < 5 and \
                                exs_counts[event_in_dataset] < 2:
                            exs_counts[event_in_dataset] += 1
                            self.examples[event].append((wordA, wordB))

    def evaluate_alignment(self, path, sound_model, filename):

        # specific settings to maximise compatibility

        # If we keep this to "True", sequences are re-tokenised when calling PSA
        # as the gold sequences are already merged
        lingpy.settings.rcParams["merge_vowels"] = True

        # This allows to keep sequences with sounds unknown from CLTS.
        # We just won't coarsen them.
        sound_model.silent_errors = True

        gold = lingpy.PSA(path) # used for evaluation
        pred = lingpy.PSA(path) # alignments will be replaced by our method
        lingpy.settings.rcParams["merge_vowels"] = False

        for i, (seqA, seqB) in enumerate(pred.tokens):
            # Replace sequences with coarsened sequences
            seqA = [sound_model[s] for s in seqA]
            seqB = [sound_model[s] for s in seqB]
            gold.tokens[i] = (seqA, seqB)
            pred.tokens[i] = (seqA, seqB)

            # Replace gold data with coarsened sequences
            goldA, goldB, goldSim = gold.alignments[i]
            goldA = [s if s =="-" else sound_model[s] for s in goldA]
            goldB = [s if s =="-" else sound_model[s] for s in goldB]
            gold.alignments[i] = goldA, goldB, goldSim

            # Perform alignment on coarse sequences
            pred.alignments[i] = self.align(seqA, seqB)

        eval = lingpy.evaluate.apa.EvalPSA(gold, pred)
        scores = {"Column score (BDPA)":eval.c_score(),
                  "Sum of pair score (BDPA)": eval.sp_score(),
                  "Jaccard score (BDPA)": eval.jc_score(),
                  "Percentage of identical rows (BDPA)": eval.r_score(),
                  }
        eval.diff(filename=filename)
        sound_model.silent_errors = False
        return scores


def run(args):
    """Run the correspondence command.

    Run with:

        cldfbench lexitools.correspondences --clts-version v1.4.1 --model Coarse --cutoff 0.05 --threshold 1 --dataset lexicore

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
    langgenera_path = "./src/lexitools/commands/lang_genera-v1.0.0.tsv"
    clts = args.clts.from_config().api
    now = time.strftime("%Y%m%d-%Hh%Mm%Ss")
    output_prefix = "{timestamp}_sound_correspondences".format(timestamp=now)

    """Three options for sound classes:
    
    1. Keep original BIPA symbols. Very precise, but variations in annotation make the results noisy.
    2. ASJP. Only possible with the lexibank/asjp dataset, as we don't have any BIPA->ASJPcode converter.
    3. Coarsen BIPA to keep only some main features. This tries to strike a balance between BIPA and avoiding noise.
    """
    full_asjp = False
    if args.model == "BIPA":
        def to_sound_class(sound):
            return str(clts.bipa[sound])
    elif args.model == "Coarse":
        coarse = Coarsen(clts.bipa, "src/lexitools/commands/default_coarsening.csv")

        def to_sound_class(sound):
            return coarse[sound]
    elif args.model == "ASJPcode":
        if args.dataset != "lexibank/asjp":
            raise ValueError("ASJPcode only possible with lexibank/asjp")
        full_asjp = True

        def to_sound_class(sound):
            return sound  # we will grab the graphemes column
    else:
        raise ValueError("Incorrect sound class model")

    ## This is a temporary fake "lexicore" interface
    dataset_list = LEXICORE if args.dataset == "lexicore" else [args.dataset.split("/")]
    db = MockLexicore(dataset_list)

    data = SoundCorrespsByGenera(db, langgenera_path,
                                 sound_class=to_sound_class,
                                 concept_list=args.concepts,
                                 glottolog=pyglottolog.Glottolog(
                                     args.glottolog.dir),
                                 asjp=full_asjp)

    args.log.info(
        'Loaded the wordlist ({} languages, {} genera, {} concepts kept)'.format(
            len(data.lang_to_concept),
            len(data.genera_to_lang),
            len(data.concepts_subset)))

    corresp_finder = Correspondences(args, data, clts,
                                     align_method=args.alignment)

    align_eval = {}
    if args.model == "Coarse" and args.bdpa is not None:
        align_eval = corresp_finder.evaluate_alignment(args.bdpa,
                                      coarse,
                                      output_prefix + '_alignment_eval.diff')

    available = corresp_finder.find_available()

    # with cProfile.Profile() as pr:
    # TODO: if this needs to be much more memory efficient,
    # we can work genus-per-genus, write to file as we work,
    # and clear memory after each genus. For now it does not seem
    # necessary.
    corresp_finder.find_attested_corresps()

    # pr.dump_stats("profile.prof")

    def format_ex(rows):
        r1, r2 = rows
        tok1 = " ".join(r1.token)
        tok2 = " ".join(r2.token)
        template = "{word} ([{original_token}] {dataset} {id})"

        return template.format(word=tok1, **asdict(r1)) + "/" + \
               template.format(word=tok2, **asdict(r2))

    with open(output_prefix + '_coarsening.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerows(coarse.as_table())

    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Lang A", "Lang B", "Sound A",
                         "Sound B", "Env A", "Env B", "Count", "Examples"])
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
    with open(output_prefix + '_pairs_count_by_datasets.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["DatasetA", "DatasetB", "Number_of_cognate_pairs"])
        for (dA, dB) in corresp_finder.cognates_pairs_by_datasets:
            count =corresp_finder.cognates_pairs_by_datasets[(dA, dB)]
            writer.writerow([dA, dB, count])
            if dA == dB:
                pairs_in_dataset += count
            else:
                pairs_across_datasets += count

    with open(output_prefix + '_available.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Sound A", "Sound B"])
        writer.writerows(available)

    with open(output_prefix + '_concepts_intersection.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Lang A", "Lang B", "Common concepts", "Kept concepts"])
        for lA, lB in data.concepts_intersection:
            concepts = data.concepts_intersection[(lA, lB)]
            kept = corresp_finder.total_cognates[(lA, lB)]
            writer.writerow([lA.glottocode, lB.glottocode, concepts, kept])

    metadata_dict = {"observation cutoff": args.cutoff,
                     "similarity threshold": args.threshold,
                     "model": args.model,
                     "concepts": args.concepts,
                     "dataset": args.dataset}
    metadata_dict.update(align_eval)
    dataset_str = sorted(a + "/" + b for a, b in dataset_list)
    metadata_dict["dataset_list"] = dataset_str
    metadata_dict["n_languages"] = len(data.lang_to_concept)
    metadata_dict["n_genera"] = len(data.genera_to_lang)
    metadata_dict["n_concepts"] = len(data.concepts_subset)
    metadata_dict["n_tokens"] = len(data.data)
    metadata_dict["n_cognate_pairs_across_datasets"] = pairs_across_datasets
    metadata_dict["n_cognate_pairs_in_datasets"] = pairs_in_dataset
    metadata_dict["threshold_method"] = "normalized per syllable"
    metadata_dict["cutoff_method"] = "max(2, cutoff * shared_cognates)"
    metadata_dict["alignment_method"] = args.alignment

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")