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

from itertools import combinations
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pylexibank import progressbar

import networkx as nx
import numpy as np
import csv


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


def iter_phonemes(tokens):
    for segment in tokens:
        if "/" in segment:
            segment = segment.split("/")[1]
        yield segment


def get_langgenera_mapping(path):
    langgenera = {}
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)
        for row in reader:
            row_dict = dict(zip(header, row))
            langgenera[row_dict["GLOTTOCODE"]] = row_dict["GENUS"]
    return langgenera


def sound_corresp_matrix(G, attr="total freq"):
    matrix = nx.to_numpy_matrix(G, weight=attr)
    return matrix.tolist()


def run(args):
    ds = get_dataset(args)
    clts = args.clts.from_config().api
    if args.model == "bipa":
        sound_class = clts.transcriptionsystem_dict[args.model]
    else:
        sound_class = clts.soundclasses_dict[args.model]

    genera = get_langgenera_mapping(
        "./src/lexitools/commands/lang_genera-v1.0.0.tsv")

    concepticon = Concepticon(Config.from_file().get_clone("concepticon"))
    concept_dict = concepticon.conceptlists[args.concepts].concepts
    concepts = {c.concepticon_id for c in concept_dict.values() if
                c.concepticon_id}

    # columns=('parameter_id', 'concept_name', 'language_id', 'language_name',
    #         'value', 'form', 'segments', 'language_glottocode',
    #         'concept_concepticon_id', 'language_latitude',
    #         'language_longitude')
    # namespace=(('concept_name', 'concept'), ('language_id', 'doculect'),
    #         ('segments', 'tokens'), ('language_glottocode', 'glottolog'),
    #         ('concept_concepticon_id', 'concepticon'), ('language_latitude',
    #             'latitude'), ('language_longitude', 'longitude'), ('cognacy',
    #                 'cognacy'), ('cogid_cognateset_id', 'cogid'))

    lex = lingpy.LexStat.from_cldf(ds.cldf_dir.joinpath('cldf-metadata.json'),
                                   filter=lambda row: row[
                                                          'concept_concepticon_id'] in concepts)
    args.log.info('Loaded the wordlist')

    concepts_kept = set(
        (a, b) for idx, a, b in lex.iter_rows('concept', 'concepticon'))
    args.log.info("A total of {} concepts were used".format(len(concepts_kept)))

    available = {}
    attested = {}

    # genus -> sound -> language -> count
    sounds_by_genera = defaultdict(lambda: defaultdict(lambda: Counter()))

    args.log.info('Counting available corresp...')
    G = nx.Graph()
    for idx, tokens, language in lex.iter_rows('tokens', 'glottolog'):
        genus = genera[language]
        for sound in iter_phonemes(tokens):
            if sound not in "+*-":
                sound = sound_class[sound]
                sounds_by_genera[genus][sound][language] += 1

    available = defaultdict(list)
    for genus in sounds_by_genera:
        freq_data = sounds_by_genera[genus]
        n_sounds = len(freq_data)
        tot_sound_pairs = (n_sounds * (n_sounds - 1)) / 2
        sound_pairs = combinations(freq_data, r=2)
        for sound_A, sound_B in progressbar(sound_pairs, total=tot_sound_pairs):
            occ_A = set(
                {lg for lg in freq_data[sound_A] if freq_data[sound_A][lg] > 1})
            occ_B = set(
                {lg for lg in freq_data[sound_B] if freq_data[sound_B][lg] > 1})
            # Available if there are 2 distinct languages with each at least 2 occurences
            if occ_A and occ_B and (
                    len(occ_A) > 1 or len(occ_B) > 1 or occ_A != occ_B):
                available[genus].append((sound_A, sound_B))

    # for node, data in G.nodes(data=True):
    #     data['language'] = ', '.join(sorted(data['language']))

    args.log.info('Counting attested corresp...')
    n_lang = len(lex.cols)
    tot_pairs = (n_lang * (n_lang - 1)) / 2
    # Record correspondences
    for lA, lB in progressbar(combinations(lex.cols, r=2), total=tot_pairs):
        for idxA, idxB in lex.pairs[lA, lB]:
            tokensA = list(iter_phonemes(lex[idxA, 'tokens']))
            tokensB = list(iter_phonemes(lex[idxB, 'tokens']))
            langA, langB = lex[idxA, 'glottolog'], lex[idxB, 'glottolog']
            # classesA, classesB = lex[idxA, 'classes'], lex[idxB, 'classes']

            # check for edit dist == 1giot
            token_strs = ' '.join(tokensA), ' '.join(tokensB)
            potential_corresp = lingpy.edit_dist(*token_strs) <= args.threshold
            if potential_corresp:
                pair = lingpy.Pairwise(*token_strs)
                pair.align()
                almA, almB, sim = pair.alignments[0]
                for sound_pair in zip(almA, almB):
                    soundA, soundB = sound_pair
                    if soundA != soundB and \
                            not '-' in sound_pair and not '+' in sound_pair:
                        soundA = sound_class[soundA]
                        soundB = sound_class[soundB]
                        try:
                            G[soundA][soundB]['frequency'][(langA, langB)] += 1
                        except KeyError:
                            G.add_edge(soundA, soundB,
                                       frequency=Counter({(langA, langB): 1}))

    args.log.info('Filtering out rare corresps...')
    del_edges = []
    for nA, nB, data in G.edges(data=True):
        for lA, lB in list(data['frequency']):
            freq = data['frequency'][(lA, lB)]
            if freq < args.cutoff:
                del data['frequency'][(lA, lB)]
        if len(data['frequency']) == 0:
            del_edges += [(nA, nB)]
        else:
            freq_edge = sum(data['frequency'].values())
            data['num pairs'] = len(data['frequency'])
            data['total freq'] = freq_edge
            data['weight'] = data['num pairs'] / tot_pairs
    G.remove_edges_from(del_edges)

    #
    # table = []
    # for nA, nB, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight'],
    #                            reverse=True):
    #     data['language'] = ', '.join(
    #         ['{0}/{1}'.format(a, b) for a, b in sorted(data['frequency'])])
    #
    #     if data['total freq'] >= args.cutoff:
    #         table += [[nA, G.node[nA]['total freq'],
    #                    nB, G.node[nB]['total freq'],
    #                    data['total freq'], data['num pairs'],
    #                    data['weight']]]

    output_prefix = "{dataset}_{cutoff}occ_{threshold}cols_{model}".format(
        **vars(args))

    # Show graph
    pos = nx.spring_layout(G, iterations=60)
    weights = [G[u][v]['weight'] * 50 for u, v in G.edges()]
    nx.draw(G, pos=pos, with_labels=True,
            width=weights, node_color="#45aaf2",
            node_size=500)
    plt.savefig(output_prefix+"_graph.png",  bbox_inches="tight", pad_inches=0.1)


    # Output genus level info
    matrix = sound_corresp_matrix(G, attr="total freq")
    with open(output_prefix + '_matrix.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(list(G.nodes()))
        for r in matrix:
            writer.writerow(r)

    # Output genus level info
    with open(output_prefix + '_results.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(["Genus","Sound A","Sound B", "Availability", "Freq"])
        for genus in available:
            for a, b in available[genus]:
                try:
                    langs = G[a][b]["frequency"]
                    freq = sum([1 for lg1,lg2 in langs
                                if genera[lg1] == genus and genera[lg2] == genus])
                except KeyError:
                    freq = 0
                writer.writerow([genus, a, b, "True", freq])

    # with Table(
    #         args,
    #         *['Sound A', 'Freq A', 'Sound B', 'Freq B', 'Occ', 'Pairs', 'Weight'],
    #         rows=sorted(table, key=lambda x: (str(x[0]), str(x[2])))):
    #     pass
    #
    #
    # IG = networkx2igraph(G)
    # im = IG.community_infomap()
    # for i, comm in enumerate(im):
    #     for node in comm:
    #         IG.vs[node]['infomap'] = i+1
    #
    # IG.write_gml('{0}-graph.gml'.format(args.dataset))
