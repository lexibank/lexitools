"""
Extract correspondences from a set of datasets.
"""

from clldutils.clilib import add_format
from cldfbench.cli_util import add_catalog_spec

from pyconcepticon.api import Concepticon
from cldfcatalog import Config

import lingpy
from itertools import combinations, product
from collections import Counter, defaultdict, namedtuple
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
            ('sequencecomparison', 'zhivlovobugrian')]


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
        needed_install = False
        for org, name in dataset_list:
            if find_spec("lexibank_" + name) is None:
                args = [sys.executable, "-m", "pip", "install",
                        "-e", egg_pattern.format(org=org, name=name)]
                subprocess.run(args)
                needed_install = True

        if needed_install:
            raise EnvironmentError("Some datasets were not installed. ",
                                   "I have tried to install them,",
                                   "please re-run the command now.")

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


class SoundCorrespsByGenera(object):
    """ Loads lexibank data, organized by genera, to facilitate the counting of correspondences.

    Attributes:
        sound_class (func): a function which associates a class of sounds to any BIPA
            sound string.
        lang_to_genera (dict): a mapping of glottocodes (representing languages) to
            genera identifiers. This is loaded from Tiago Tresoldi's langgenera.
        genera_to_lang (defaultdict): the inverse mapping, only for languages encountered
            in our data.
        genera_to_family (dict): mapping from genera names to family names for genera
            encountered in our data.
        errors (list of list): table which summarizes errors encountered in tokens.
        asjp (bool): whether the only dataset is ASJP, in which case we will extract
            Graphemes rather than Segments.
        concepts_intersection (Counter): Counts the number of distinct concepts kept
            for each pair of languages encountered in a genera.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of glottocodes to concepts.
        tokens (defaultdict): mapping of (glottocode, concept) to a set of tokens for this
            language and concept. While there is usually only one, there may be more.
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
            self.lang_to_genera = {row['GLOTTOCODE']: row['GENUS'] for row in reader}

        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        langs = {}
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        self.genera_to_lang = defaultdict(set)
        self.genera_to_family = {}
        self.errors = [["Dataset", "Language_ID", "Sound", "Token", "ID"]]
        self.asjp = asjp
        self.concepts_intersection = Counter()

        for row in progressbar(db.iter_table("LanguageTable"),
                               desc="listing languages"):
            family = row["Family"]
            gcode = row["Glottocode"]
            if gcode is None or gcode not in self.lang_to_genera:
                continue
            id = row["ID"]
            langoid = langoids.get(gcode, None)
            if langoid is not None and langoid.family is not None:
                family = langoid.family.name

            genus = self.lang_to_genera[gcode]
            self.genera_to_lang[genus].add(gcode)
            if genus not in self.genera_to_family:
                self.genera_to_family[genus] = family
            langs[id] = (gcode, family, genus)

        concepts = {row["ID"]: row["Concepticon_ID"] for row in
                    db.iter_table('ParameterTable')}

        if concept_list is not None:
            concepticon = Concepticon(
                Config.from_file().get_clone("concepticon"))
            concept_dict = concepticon.conceptlists[concept_list].concepts
            self.concepts_subset = {c.concepticon_id for c in
                                    concept_dict.values() if
                                    c.concepticon_id}
        else:  # All observed concepts
            self.concepts_subset = set(concepts.values())

        self.lang_to_concept = defaultdict(set)
        self.tokens = defaultdict(set)

        for row in progressbar(db.iter_table('FormTable'),
                               desc="Loading data..."):
            concept = concepts[row["Parameter_ID"]]
            if concept not in self.concepts_subset or \
                    row["Language_ID"] not in langs or row["Segments"] is None:
                continue

            try:
                token = list(self._iter_phonemes(row))
            except ValueError:
                continue  # unknown sounds
            if token == [""]: continue
            syllables = len(
                lingpy.sequence.sound_classes.syllabify(token, output="nested"))
            gcode, family, genus = langs[row["Language_ID"]]
            self.tokens[(gcode, concept)].add(
                (tuple(token), syllables))  ## Create a token object or row object.
            self.lang_to_concept[gcode].add(concept)

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
                    tokens_A = self.tokens[(langA, concept)]
                    tokens_B = self.tokens[(langB, concept)]
                    for (tA, sA), (tB, sB) in product(tokens_A, tokens_B):
                        yield genus, (langA, tA, sA), (langB, tB, sB)

    def __iter__(self):
        """Iterate over the tokens.

        Yields:
            for all known tokens, its genus, language glottocode, concept, and the token itself.
        """
        for lang, concept in self.tokens:
            genus = self.lang_to_genera[lang]
            for token, syll_count in self.tokens[(lang, concept)]:
                yield genus, lang, concept, token


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
        sca (dict): mapping of bipa sounds to SCA class (used for the cognate threshold).
        Item (namedtuple): A sound observed in a specific context and language.
         Each correspondence is made of a pair of items.
        corresps (Counter): counts occurences of pairs of Items (the keys are frozensets).
        total_cognates (Counter): counts the number of cognates found for each pair of languages.
        ignore (set): set of characters which should be ignored in tokens (markers, etc)
    """

    def __init__(self, args, data, clts):
        """ Initialization only records the arguments and defines the attributes.

        Args:
            args: the full args passed to the correspondences command.
            data (SoundCorrespsByGenera): the lexicore dataset
            clts (pyclts.CLTS): a clts instance
        """
        self.args = args
        self.data = data
        self.clts = clts
        self.sca = self.clts.soundclasses_dict["sca"]
        self.corresps = Counter()  # frozenset({Item,Item}): count
        self.total_cognates = Counter()  # (lgA, lgB) : count
        self.Item = namedtuple("Item", ["genus", "lang", "sound", "context"])
        self.ignore = set("+*#_")

    def find_available(self):
        """ Find which pairs of sounds from our data are available in each genera.

        - A pair of two distinct sounds x,y are available in a genus if the genus has at
         least two distinct languages A,B such that A has at least two occurences of x
         and B has at least two occurences of y.
        - A pair of a sound and a gap (x,-) is available in a genus if that genus has at
         least two occurences of x.
        - A pair of a sound and itself (x,x) is available in a genus if that genus has at
         least two occurences of x.

        Returns:
            available (defaultdict): maps a pair of sounds (str) to a list of genera
                in which the sound is available.
        """
        # Available if there are 2 distinct languages with each at least 2 occurences

        self.args.log.info('Counting available corresp...')

        sounds_by_genera = defaultdict(lambda: defaultdict(Counter))
        for genus, lang, concept, token in self.data:
            for sound in token:
                if sound not in "+*#_":  # spaces and segmentation symbols ignored
                    sounds_by_genera[genus][sound][lang] += 1

        available = defaultdict(list)
        for genus in sounds_by_genera:
            freq = sounds_by_genera[genus]
            n_sounds = len(freq)
            tot_sound_pairs = (n_sounds * (n_sounds - 1)) / 2
            sound_pairs = combinations(freq, r=2)

            for sound_A in progressbar(freq):
                if sound_A != "-":  # No point in counting corresp between blank and itself
                    occ = {lg for lg in freq[sound_A] if freq[sound_A][lg] > 1}
                    if len(occ) > 1:
                        available[(sound_A, sound_A)].append(genus)
                    if len(occ) > 0:
                        available[(sound_A, "-")].append(
                            genus)  # Deletion of any available sound is available

            for sound_A, sound_B in progressbar(sound_pairs, total=tot_sound_pairs):
                occ_A = {lg for lg in freq[sound_A] if freq[sound_A][lg] > 1}
                occ_B = {lg for lg in freq[sound_B] if freq[sound_B][lg] > 1}
                if occ_A and occ_B and len(occ_A | occ_B) > 1:
                    sound_pair = tuple(sorted((sound_A, sound_B)))
                    available[sound_pair].append(genus)

        return available

    def differences(self, ta, tb):
        """ Count meaningful differences between two tokens.

        We take the edit distance between the sound classes in each token.
        This allows any changes insides SCA's classes for free, that is, expected changes
        are not penalized.

        Args:
            ta (list of str): a token
            tb (list of str): another token

        Returns:
            diff (int): the edit distance between the sound classes in each token.
        """
        ta = " ".join([self.sca[s] for s in ta])
        tb = " ".join([self.sca[s] for s in tb])
        return lingpy.edit_dist(ta, tb)

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

    def iter_with_contexts(self, almA, almB):
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
                cat = self.clts.bipa[s].type
                if s == "-" or s in self.ignore or cat in {"tone"}:
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

        Here, we set 1 for a match, -10 for mismatches involving a tone and something
        that is not a t one, and -1 for any other mismatches and indels.

        The reason to penalize tones is that having tones in the sequence of sounds is
        only a notational trick, as they actually belong to a different tier.

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
            ta = self.clts.bipa[a].type == "tone"
            tb = self.clts.bipa[b].type == "tone"
            if ta != tb:
                return -10
            else:
                return -1

        return {(a, b): score(a, b) for a, b in product(seqA, seqB)}

    def find_attested_corresps(self):
        """ Find all correspondences attested in our data.

        - Inside each genus, we consider all pairs of tokens for the same concept across two
        languages.
        - If the tokens are similar enough, we assume that they are cognates. Otherwise
         we reject the pair.
        - We then align cognate pairs using our custom scorer.
        - We record a correspondence for each aligned position, unless a sound is to be ignored.

        We do not use a better cognate recognition method or cognate alignment function,
        as these tend to insert too much knowledge, which we then find back identical in
        correspondences.

        TODO: (maybe)
            - We do not yet align partial cognates separately.
            - We do not yet use the gold cognate sets.

        This functions returns None, but changes `self.corresps` in place.
        """
        self.args.log.info('Counting attested corresp...')
        for genus, (lA, tokensA, s_A), (lB, tokensB, s_B) in self.data.iter_candidates():
            dist = self.differences(tokensA, tokensB)
            allowed = self.allowed_differences(s_A, s_B)
            if dist <= allowed:
                self.total_cognates[(lA, lB)] += 1
                almA, almB, sim = lingpy.nw_align(tokensA, tokensB,
                                                  self.get_scorer(tokensA, tokensB))
                for (sA, cA), (sB, cB) in self.iter_with_contexts(almA, almB):
                    if self.ignore.isdisjoint({sA, sB}):
                        A = self.Item(genus=genus, lang=lA, sound=sA, context=cA)
                        B = self.Item(genus=genus, lang=lB, sound=sB, context=cB)
                        self.corresps[frozenset({A, B})] += 1


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
        tokens have been ignored by the program. The headers for this table is
        `Dataset,Language_ID,Sound,Token,ID`.
    `_metadata.json`: a json file recording the input parameters and all relevant metadata.
    """
    langgenera_path = "./src/lexitools/commands/lang_genera-v1.0.0.tsv"
    clts = args.clts.from_config().api

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

    corresp_finder = Correspondences(args, data, clts)
    available = corresp_finder.find_available()
    corresp_finder.find_attested_corresps()

    now = time.strftime("%Y%m%d-%Hh%Mm%Ss")

    output_prefix = "{timestamp}_sound_correspondences".format(timestamp=now)

    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Lang A", "Lang B", "Sound A",
                         "Sound B", "Env A", "Env B", "Count"])
        for sounds in corresp_finder.corresps:
            A, B = sorted(sounds,
                          key=lambda s: s.sound)  # ensures we always have the same order.
            count = corresp_finder.corresps[sounds]
            family = data.genera_to_family[A.genus]
            total = corresp_finder.total_cognates[(A.lang, B.lang)]
            if count > max(2, args.cutoff * total):
                writer.writerow([family, A.genus, A.lang, B.lang,
                                 A.sound, B.sound, A.context, B.context, count])

    with open(output_prefix + '_available.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Sound A", "Sound B"])
        for a, b in available:
            for genus in available[(a, b)]:
                writer.writerow([data.genera_to_family[genus], genus, a, b])

    with open(output_prefix + '_concepts_intersection.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Lang A", "Lang B", "Common concepts", "Kept concepts"])
        for lA, lB in data.concepts_intersection:
            concepts = data.concepts_intersection[(lA, lB)]
            kept = corresp_finder.total_cognates[(lA, lB)]
            writer.writerow([lA, lB, concepts, kept])

    metadata_dict = {"observation cutoff": args.cutoff,
                     "similarity threshold": args.threshold,
                     "model": args.model,
                     "concepts": args.concepts,
                     "dataset": args.dataset}
    dataset_str = sorted(a + "/" + b for a, b in dataset_list)
    metadata_dict["dataset_list"] = dataset_str
    metadata_dict["n_languages"] = len(data.lang_to_concept)
    metadata_dict["n_genera"] = len(data.genera_to_lang)
    metadata_dict["n_concepts"] = len(data.concepts_subset)
    metadata_dict["n_tokens"] = len(data.tokens)
    metadata_dict["threshold_method"] = "normalized per syllable"
    metadata_dict["cutoff_method"] = "max(2, cutoff * shared_cognates)"
    metadata_dict["alignment_method"] = "T/non T penalized"

    ## TODO: export 5 examples for each sound
    if args.model == "Coarse":
        with open(output_prefix + '_coarsening.csv', 'w',
                  encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',', )
            writer.writerows(coarse.as_table())

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")
