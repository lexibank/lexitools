"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import add_format
from cldfbench.cli_util import add_catalog_spec

from pyconcepticon.api import Concepticon
from cldfcatalog import Config

# lingpy for alignments
import lingpy

from itertools import combinations, product
from collections import Counter, defaultdict
from pylexibank import progressbar
import pyglottolog
import networkx as nx
import csv
import csvw
import time
import subprocess
from importlib.util import find_spec
from cldfbench import get_dataset
import sys

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
                 sound_class=None, glottolog=None):
        self.sound_class = sound_class
        self.lang_to_genera = get_langgenera_mapping(langgenera_path)

        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        langs = {}
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        self.genera_to_lang = defaultdict(set)
        self.genera_to_family = {}

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
        self.tokens = defaultdict(list)

        for row in progressbar(db.iter_table('FormTable'),
                               desc="Loading data..."):
            concept = concepts[row["Parameter_ID"]]
            if concept not in self.concepts_subset or \
                    row["Language_ID"] not in langs or row["Segments"] is None:
                continue
            try:
                token = list(self._iter_phonemes(row["Segments"]))
            except KeyError as e:
                continue  # skip this token, it contains unknown sounds
            if token == [""]: continue

            gcode, family, genus = langs[row["Language_ID"]]
            self.tokens[(gcode, concept)].append(token)
            self.lang_to_concept[gcode].add(concept)

        self.concepts_intersection = []
        log.info(r"Total number of concepts kept: {}".format(
            len(self.concepts_subset)))

    def _iter_phonemes(self, segments):
        # In some dataset, the separator defined in the metadata is " + ",
        # which means that tokens are not phonemes (ex:bodtkhobwa)
        # This is solved by re-tokenizing on the space...
        tokens = " ".join([s for s in segments if s is not None]).split(" ")
        for segment in tokens:
            if "/" in segment:
                segment = segment.split("/")[1]
            yield self.sound_class(segment)

    def iter_candidate_pairs(self, genus):
        langs = self.genera_to_lang[genus]
        lang_pairs = combinations(langs, r=2)
        n_lang = len(langs)
        tot_pairs = (n_lang * (n_lang - 1)) / 2
        for langA, langB in progressbar(lang_pairs, total=tot_pairs,
                                        desc="Language pairs"):
            concepts_A = self.lang_to_concept[langA]
            concepts_B = self.lang_to_concept[langB]
            common_concepts = (concepts_A & concepts_B)
            self.concepts_intersection.append((genus, langA, langB,
                                               len(common_concepts)))
            for concept in common_concepts:
                tokens_A = self.tokens[(langA, concept)]
                tokens_B = self.tokens[(langB, concept)]
                for tA, tB in product(tokens_A, tokens_B):
                    yield (langA, tA), (langB, tB)

    def __iter__(self):
        for lang, concept in self.tokens:
            genus = self.lang_to_genera[lang]
            for token in self.tokens[(lang, concept)]:
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
        type=int,
        help='select a threshold')
    parser.add_argument(
        '--cutoff',
        action='store',
        default=2,
        type=float,
        help='select a threshold')

    parser.add_argument(
        '--model',
        action='store',
        default='bipa',
        type=str,
        help='select a sound class model: asjp, bipa, color, sca')

    parser.add_argument(
        '--concepts',
        action='store',
        default=None,
        type=str,
        help='select a concept list to filter on')

    ## TODO: add back an option to run on single dataset -> in order to allow for exact ASJP replication.


def get_langgenera_mapping(path):
    with csvw.UnicodeDictReader(path, delimiter="\t") as reader:
        return {row['GLOTTOCODE']: row['GENUS'] for row in reader}


def find_available(args, data):
    # Available if there are 2 distinct languages with each at least 2 occurences

    args.log.info('Counting available corresp...')

    sounds_by_genera = defaultdict(lambda: defaultdict(lambda: Counter()))
    for genus, lang, concept, token in data:
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


def find_attested_corresps(args, data, is_bipa=False):
    args.log.info('Counting attested corresp...')
    G = nx.Graph()
    ignore = "+*#_"
    for genus in progressbar(data.genera_to_lang, desc="Genera"):
        for (lA, tokensA), (lB, tokensB) in data.iter_candidate_pairs(genus):
            tokens = (' '.join(tokensA), ' '.join(tokensB))
            if lingpy.edit_dist(*tokens) <= args.threshold:
                if is_bipa:
                    alignments = lingpy.Pairwise(*tokens)()
                    almA, almB, sim = alignments[0] # Skipping any other alignments
                else:
                    almA, almB, sim = lingpy.nw_align(*tokens)
                for sound_pair in zip(almA, almB):
                    soundA, soundB = sound_pair
                    if soundA not in ignore and \
                            soundB not in ignore and \
                            (soundA != "-" or soundB != "-"):
                        try:
                            G[soundA][soundB]['occurences'][(lA, lB)] += 1
                        except KeyError:
                            G.add_edge(soundA, soundB,
                                       occurences=Counter({(lA, lB): 1}))

    args.log.info('Filtering out rare corresps...')
    del_edges = []
    for nA, nB in G.edges():
        for lA, lB in list(G[nA][nB]['occurences']):
            freq = G[nA][nB]['occurences'][(lA, lB)]
            if freq < args.cutoff:
                del G[nA][nB]['occurences'][(lA, lB)]
        if len(G[nA][nB]['occurences']) == 0:
            del_edges += [(nA, nB)]
        else:
            G[nA][nB]['weight'] = len(
                G[nA][nB]['occurences'])  # number of lg pair
    G.remove_edges_from(del_edges)
    return G


def run(args):
    langgenera_path = "./src/lexitools/commands/lang_genera-v1.0.0.tsv"
    clts = args.clts.from_config().api
    if args.model == "bipa":
        def sound_class(sound):
            sound_system = clts.transcriptionsystem_dict[args.model]
            return str(sound_system[sound])
    else:
        def sound_class(sound):
            sound_system = clts.soundclasses_dict[args.model]
            extras = {'cʃ': 'TS~', 'pʃ': 'pS~', 'pχʼ': 'p"X~',
                      'd̪ʒ': 'j'}  # these are missing from the clts model
            try:
                return str(sound_system[sound])
            except KeyError:
                return extras[sound]

    ## This is a temporary fake "lexicore" interface
    if args.dataset == "lexicore":
        dataset_list = [('lexibank', 'aaleykusunda'),
                        ('lexibank', 'abrahammonpa'), ('lexibank', 'allenbai'),
                        ('lexibank', 'bdpa'), ('lexibank', 'beidasinitic'),
                        ('lexibank', 'birchallchapacuran'),
                        ('sequencecomparison', 'blustaustronesian'),
                        ('lexibank', 'bodtkhobwa'), ('lexibank', 'bowernpny'),
                        ('lexibank', 'cals'), ('lexibank', 'castrosui'),
                        ('lexibank', 'castroyi'),
                        ('lexibank', 'chaconarawakan'),
                        ('lexibank', 'chaconbaniwa'),
                        ('lexibank', 'chaconcolumbian'),
                        ('lexibank', 'chenhmongmien'),
                        ('lexibank', 'chindialectsurvey'),
                        ('lexibank', 'davletshinaztecan'),
                        ('lexibank', 'deepadungpalaung'),
                        ('lexibank', 'dravlex'), ('lexibank', 'dunnaslian'),
                        ('sequencecomparison', 'dunnielex'), ('lexibank', 'galuciotupi'),
                        ('lexibank', 'gerarditupi'), ('lexibank', 'halenepal'),
                        ('lexibank', 'hantganbangime'),
                        ('sequencecomparison', 'hattorijaponic'),
                        ('sequencecomparison', 'houchinese'),
                        ('lexibank', 'hubercolumbian'),
                        ('lexibank', 'ivanisuansu'),
                        ('lexibank', 'johanssonsoundsymbolic'),
                        ('lexibank', 'joophonosemantic'),
                        ('sequencecomparison', 'kesslersignificance'),
                        ('lexibank', 'kraftchadic'),
                        ('lexibank', 'leekoreanic'),
                        ('lexibank', 'lieberherrkhobwa'),
                        ('lexibank', 'lundgrenomagoa'),
                        ('lexibank', 'mannburmish'),
                        ('lexibank', 'marrisonnaga'),
                        ('lexibank', 'mcelhanonhuon'),
                        ('lexibank', 'mitterhoferbena'),
                        ('lexibank', 'naganorgyalrongic'),
                        ('lexibank', 'northeuralex'),
                        ('lexibank', 'peirosaustroasiatic'),
                        ('lexibank', 'pharaocoracholaztecan'),
                        ('lexibank', 'robinsonap'), ('lexibank', 'sagartst'),
                        ('lexibank', 'savelyevturkic'),
                        ('lexibank', 'sohartmannchin'),
                        ('sequencecomparison', 'starostinpie'), ('lexibank', 'suntb'),
                        ('lexibank', 'transnewguineaorg'),
                        ('lexibank', 'tryonsolomon'),
                        ('lexibank', 'walkerarawakan'),
                        ('lexibank', 'walworthpolynesian'),
                        ('lexibank', 'wold'), ('lexibank', 'yanglalo'),
                        ('lexibank', 'zgraggenmadang'), ('lexibank', 'zhaobai'),
                        ('sequencecomparison', 'zhivlovobugrian')]

    else:
        dataset_list = [args.dataset.split("/")]

    db = MockLexicore(dataset_list)
    data = SoundCorrespsByGenera(db, langgenera_path, args.log,
                                 sound_class=sound_class,
                                 concept_list=args.concepts,
                                 glottolog=pyglottolog.Glottolog(
                                     args.glottolog.dir))

    args.log.info('Loaded the wordlist ({} languages, {} genera)'.format(
        len(data.lang_to_concept),
        len(data.genera_to_lang)))

    available = find_available(args, data)

    G = find_attested_corresps(args, data, is_bipa=args.model == "bipa")

    now = time.strftime("%Y%m%d-%Hh%M")

    output_prefix = "{timestamp}_corresp_min_occ_{cutoff}_" \
                    "min_sim_{threshold}_sounds_{model}_data_{dataset}_concepts_{concepts}".format(
        timestamp=now,
        **vars(args)).replace("/","-")

    # Output genus level info
    with open(output_prefix + '_results.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Family", "Genus", "Sound A",
                         "Sound B", "Available", "Observed"])
        for a, b in available:
            occ_by_genus = Counter()
            if G.has_edge(a, b):
                for lA, lB in G[a][b]["occurences"]:
                    occ_by_genus[data.lang_to_genera[lA]] += \
                        G[a][b]["occurences"][(lA, lB)]

            for genus in available[(a, b)]:
                writer.writerow([data.genera_to_family[genus], genus, a, b, 1,
                                 occ_by_genus[genus]])

    with open(output_prefix + '_concepts_intersection.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Genus", "Lang A", "Lang B", "Common concepts"])
        writer.writerows(data.concepts_intersection)
