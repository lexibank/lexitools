"""
Check the prosodic structure of a given dataset.
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
from lexitools.coarse_soundclass import Coarsen, DEFAULT_CONFIG


class MockLexicore(object):
    """This is a mock to access the data without waiting on the Lexibank SQL project,
    but with a reasonable interface so that we can plug in the correct interface soon with little effort."""

    def __init__(self, dataset_list):
        # see https://github.com/chrzyki/snippets/blob/main/lexibank/install_datasets.py
        egg_pattern = "git+https://github.com/{org}/{name}.git#egg=lexibank_{name}"
        for org, name in dataset_list:
            if find_spec("lexibank_" + name) is None:
                args = [sys.executable, "-m", "pip", "install",
                        "-e", egg_pattern.format(org=org, name=name)]
                subprocess.run(args)

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
        for name, ds in self.datasets.items():
            for row in ds[table_name]:
                row["dataset"] = name
                yield row

    def get_table(self, table_name):
        return list(self.iter_table(table_name))


class SoundCorrespsByGenera(object):

    def __init__(self, db, langgenera_path, log, concept_list=None,
                 sound_class=None, glottolog=None, asjp=False):
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
            except ValueError as e:
                continue  # unknown sounds
            if token == [""]: continue
            syllables = len(
                lingpy.sequence.sound_classes.syllabify(token, output="nested"))
            gcode, family, genus = langs[row["Language_ID"]]
            self.tokens[(gcode, concept)].add((tuple(token), syllables))
            self.lang_to_concept[gcode].add(concept)

        log.info(r"Total number of concepts kept: {}".format(
            len(self.concepts_subset)))

    def _iter_phonemes(self, row):
        # In some dataset, the separator defined in the metadata is " + ",
        # which means that tokens are not phonemes (ex:bodtkhobwa)
        # This is solved by re-tokenizing on the space...
        if self.asjp:
            segments = row["Graphemes"][1:-1]
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
        for lang, concept in self.tokens:
            genus = self.lang_to_genera[lang]
            for token, syll_count in self.tokens[(lang, concept)]:
                yield genus, lang, concept, token


def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")
    add_catalog_spec(parser, "glottolog")
    add_format(parser, default='pipe')

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

    def __init__(self, args, data, clts):
        self.args = args
        self.data = data
        self.clts = clts
        self.sca = self.clts.soundclasses_dict["sca"]
        self.corresps = Counter()  # frozenset({Item,Item}): count
        self.total_cognates = Counter()  # (lgA, lgB) : count
        self.Item = namedtuple("Item", ["genus", "lang", "sound", "context"])
        self.ignore = set("+*#_")

    def find_available(self):
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
        ta = " ".join([self.sca[s] for s in ta])
        tb = " ".join([self.sca[s] for s in tb])
        return lingpy.edit_dist(ta, tb)

    def allowed_differences(self, sa, sb):
        return max(sa, sb) * self.args.threshold

    def with_contexts(self, almA, almB):
        def ctxt(sequence):
            for s in sequence:
                if s == "-" or s in self.ignore:
                    yield "-"
                elif self.clts.bipa[s].type in ["vowel", "diphthong"]:
                    yield "V"
                else:
                    yield ""
            yield "#"

        def right_context(ctxt):
            return next((c for c in ctxt if c != "-"))

        catsA = list(ctxt(almA))
        catsB = list(ctxt(almB))
        l = len(almA)
        prevA, prevB = "#", "#"
        for i in range(l):
            sA = almA[i]
            sB = almB[i]
            if catsA[i] == "-":
                cA = "-"
            else:
                cA = prevA + sA + right_context(catsA[i + 1:])
                prevA = catsA[i]
            if catsB[i] == "-":
                cB = "-"
            else:
                cB = prevB + sB + right_context(catsB[i + 1:])
                prevB = catsB[i]
            yield (sA, cA), (sB, cB)

    def get_scorer(self, seqA, seqB):
        """Custom alignment scorer which penalizes tone alignments with non tones."""
        def score(a,b):
            if a == b:
                return 1
            ta = self.clts.bipa[a].type == "tone"
            tb = self.clts.bipa[b].type == "tone"
            if ta != tb:
                # Alignments of tones with anything but tones is penalized
                # as having tones in the sequence is only a notational trick
                return -10
            else:
                return -1
        return {(a, b):score(a,b) for a, b in product(seqA, seqB)}

    def find_attested_corresps(self):
        self.args.log.info('Counting attested corresp...')
        for genus, (lA, tokensA, s_A), (lB, tokensB, s_B) in self.data.iter_candidates():
            dist = self.differences(tokensA, tokensB)
            allowed = self.allowed_differences(s_A, s_B)
            if dist <= allowed:
                self.total_cognates[(lA, lB)] += 1
                almA, almB, sim = lingpy.nw_align(tokensA, tokensB, self.get_scorer(tokensA, tokensB))
                for (sA, cA), (sB, cB) in self.with_contexts(almA, almB):
                    if self.ignore.isdisjoint({sA, sB}):
                        A = self.Item(genus=genus, lang=lA, sound=sA, context=cA)
                        B = self.Item(genus=genus, lang=lB, sound=sB, context=cB)
                        self.corresps[frozenset({A, B})] += 1


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


def run(args):
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
        coarse = Coarsen(clts.bipa, DEFAULT_CONFIG)
        def to_sound_class(sound):
            return coarse[sound]
    elif args.model == "ASJPcode":
        if args.dataset != "lexibank/asjp":
            raise ValueError("ASJPcode only possible with lexibank/asjp")
        full_asjp = True
        def to_sound_class(sound): return sound # we will grab the graphemes column

    ## This is a temporary fake "lexicore" interface
    dataset_list = LEXICORE if args.dataset == "lexicore" else [args.dataset.split("/")]

    db = MockLexicore(dataset_list)
    data = SoundCorrespsByGenera(db, langgenera_path, args.log,
                                 sound_class=to_sound_class,
                                 concept_list=args.concepts,
                                 glottolog=pyglottolog.Glottolog(
                                     args.glottolog.dir),
                                 asjp=full_asjp)

    args.log.info('Loaded the wordlist ({} languages, {} genera)'.format(
        len(data.lang_to_concept),
        len(data.genera_to_lang)))

    corresp_finder = Correspondences(args, data, clts)
    available = corresp_finder.find_available()
    corresp_finder.find_attested_corresps()

    now = time.strftime("%Y%m%d-%Hh%Mm%Ss")

    output_prefix = "{timestamp}_sound_correspondences".format(timestamp=now)

    with open(output_prefix + '_counts.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Lang a", "Lang B", "Sound A",
                         "Sound B", "Env A", "Env B", "Count"])
        for sounds in corresp_finder.corresps:
            A, B = sounds
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
    if args.model == "Coarse":
        metadata_dict["coarsening_removed"] = list(DEFAULT_CONFIG["remove"])
        metadata_dict["coarsening_changed"] = DEFAULT_CONFIG["change"]

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump(metadata_dict, metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")
