"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import Table, add_format
from cldfbench.cli_util import add_catalog_spec, get_dataset

# from linse.transform import syllable_inventories
from pylexibank.cli_util import add_dataset_spec
from pyconcepticon.api import Concepticon
from cldfcatalog import Config

# lingpy for alignments
import lingpy
# need this to store as GML format (for inspection)
from lingpy.convert.graph import networkx2igraph

from itertools import combinations, product
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pylexibank import progressbar

import networkx as nx
import csv
import csvw

class DataByGenera(object):

    def __init__(self, ds, langgenera_path, filter=None, sound_class=None):
        if filter is None:
            filter = lambda *args: True
        self.sound_class = sound_class
        self.lang_to_genera = get_langgenera_mapping(langgenera_path)
        reader = ds.cldf_reader()
        langs = {row["ID"]: row["Glottocode"] for row in
                 reader['LanguageTable']}

        concepts = {row["ID"]: row["Concepticon_ID"] for row in
                    reader['ParameterTable']}

        self.genera_to_lang = {}
        self.lang_to_concept = {}
        self.tokens = {}
        self.errors = []
        for row in ds.cldf_reader()['FormTable']:
            concept = concepts[row["Parameter_ID"]]
            lang = langs[row["Language_ID"]]
            # skip if it has no glottocode or no correct glottocode
            if lang is None or lang not in self.lang_to_genera:
                continue
            genus = self.lang_to_genera[lang]
            token = list(self._iter_phonemes(row["Segments"]))
            if filter(concept, lang, genus):
                try:
                    self.genera_to_lang[genus].add(lang)
                except:
                    self.genera_to_lang[genus] = {lang}

                try:
                    self.lang_to_concept[lang].add(concept)
                except:
                    self.lang_to_concept[lang] = {concept}

                try:
                    self.tokens[(lang, concept)].append(token)
                except:
                    self.tokens[(lang, concept)] = [token]

    def _iter_phonemes(self, tokens):
        for segment in tokens:
            if "/" in segment:
                segment = segment.split("/")[1]
            yield self.sound_class.get(segment, segment)

    def iter_candidate_pairs(self, genus):
        langs = self.genera_to_lang[genus]
        lang_pairs = combinations(langs, r=2)
        n_lang = len(langs)
        tot_pairs = (n_lang * (n_lang - 1)) / 2
        for langA, langB in progressbar(lang_pairs, total=tot_pairs):
            concepts_A = self.lang_to_concept[langA]
            concepts_B = self.lang_to_concept[langB]
            for concept in (concepts_A & concepts_B):
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

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)
    add_format(parser, default='pipe')

    # parser.add_argument(
    #        '--language-id',
    #        action='store',
    #        default=None,
    #        help='select one doculect'
    #        )
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
        default='Holman-2008-40',
        type=str,
        help='select a concept list to filter on')


def get_langgenera_mapping(path):
    with csvw.UnicodeDictReader(path, delimiter="\t") as reader:
        return {row['GLOTTOCODE']: row['GENUS'] for row in reader}

def find_available(args, data):
    args.log.info('Counting available corresp...')

    sounds_by_genera = defaultdict(lambda: defaultdict(lambda: Counter()))
    for genus, lang, concept, token in data:
        for sound in token:
            if sound not in "+*-":
                sounds_by_genera[genus][sound][lang] += 1

    available = defaultdict(list)
    for genus in sounds_by_genera:
        freq = sounds_by_genera[genus]
        n_sounds = len(freq)
        tot_sound_pairs = (n_sounds * (n_sounds - 1)) / 2
        sound_pairs = combinations(freq, r=2)
        for sound_A, sound_B in progressbar(sound_pairs, total=tot_sound_pairs):
            occ_A = {lg for lg in freq[sound_A] if freq[sound_A][lg] > 1}
            occ_B = {lg for lg in freq[sound_B] if freq[sound_B][lg] > 1}
            # Available if there are 2 distinct languages with each at least 2 occurences
            if occ_A and occ_B and \
                    (len(occ_A) > 1 or len(occ_B) > 1 or occ_A != occ_B):
                available[genus].append((sound_A, sound_B))
    return available


def make_graph(args, data):
    args.log.info('Counting attested corresp...')
    G = nx.Graph()
    for genus in progressbar(data.genera_to_lang):
        for (lA, tokensA), (lB, tokensB) in data.iter_candidate_pairs(genus):
            tokens = (' '.join(tokensA), ' '.join(tokensB))
            potential_corresp = lingpy.edit_dist(*tokens) <= args.threshold
            if potential_corresp:
                almA, almB, sim = lingpy.nw_align(*tokens)
                for sound_pair in zip(almA, almB):
                    soundA, soundB = sound_pair
                    if soundA != soundB and \
                            not '-' in sound_pair and \
                            not '+' in sound_pair:
                        try:
                            G[soundA][soundB]['frequency'][(lA, lB)] += 1
                        except KeyError:
                            G.add_edge(soundA, soundB,
                                       frequency=Counter({(lA, lB): 1}))

    args.log.info('Filtering out rare corresps...')
    del_edges = []
    for nA, nB, edge_data in G.edges(data=True):
        for lA, lB in list(edge_data['frequency']):
            freq = edge_data['frequency'][(lA, lB)]
            if freq < args.cutoff:
                del edge_data['frequency'][(lA, lB)]
        if len(edge_data['frequency']) == 0:
            del_edges += [(nA, nB)]
        else:
            edge_data['weight'] = len(edge_data['frequency'])
    G.remove_edges_from(del_edges)
    return G


def run(args):
    langgenera_path = "./src/lexitools/commands/lang_genera-v1.0.0.tsv"
    ds = get_dataset(args)
    clts = args.clts.from_config().api
    if args.model == "bipa":
        sound_class = clts.transcriptionsystem_dict[args.model]
    else:
        sound_class = clts.soundclasses_dict[args.model]

    concepticon = Concepticon(Config.from_file().get_clone("concepticon"))
    concept_dict = concepticon.conceptlists[args.concepts].concepts
    concepts = {c.concepticon_id for c in concept_dict.values() if
                c.concepticon_id}

    data = DataByGenera(ds, langgenera_path,
                        filter=lambda c, l, g: c in concepts,
                        sound_class=sound_class)

    args.log.info('Loaded the wordlist')
    concepts_kept = set.union(*data.lang_to_concept.values())
    args.log.info("A total of {} concepts were used".format(len(concepts_kept)))

    available = find_available(args, data)
    G = make_graph(args, data)

    output_prefix = "{dataset}_{cutoff}occ_{threshold}cols_{model}".format(
        **vars(args))

    # output graph
    pos = nx.spring_layout(G,k=0.5)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos=pos, with_labels=True,
            width=weights, node_color="#45aaf2",
            node_size=500)
    plt.savefig(output_prefix + "_graph.png", bbox_inches="tight",
                pad_inches=0.1)

    # Output genus level info
    with open(output_prefix + '_results.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Genus", "Sound A", "Sound B", "Availability", "Freq"])
        for genus in available:
            for a, b in available[genus]:
                try:
                    langs = G[a][b]["frequency"]
                    freq = len(langs)
                except KeyError:
                    freq = 0
                writer.writerow([genus, a, b, "True", freq])
