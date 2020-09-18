"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import Table, add_format
from cldfbench.cli_util import add_catalog_spec, get_dataset

# from linse.transform import syllable_inventories
from pylexibank.cli_util import add_dataset_spec

# lingpy for alignments
import lingpy
# need this to store as GML format (for inspection)
from lingpy.convert.graph import networkx2igraph

from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from pylexibank import progressbar

import networkx as nx


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


def iter_phonemes(tokens):
    for segment in tokens:
        if "/" in segment:
            segment = segment.split("/")[1]
        yield segment


def run(args):
    ds = get_dataset(args)
    clts = args.clts.from_config().api
    sound_class = clts.transcriptionsystem_dict[args.model]

    # columns=('parameter_id', 'concept_name', 'language_id', 'language_name',
    #         'value', 'form', 'segments', 'language_glottocode',
    #         'concept_concepticon_id', 'language_latitude',
    #         'language_longitude')
    # namespace=(('concept_name', 'concept'), ('language_id', 'doculect'),
    #         ('segments', 'tokens'), ('language_glottocode', 'glottolog'),
    #         ('concept_concepticon_id', 'concepticon'), ('language_latitude',
    #             'latitude'), ('language_longitude', 'longitude'), ('cognacy',
    #                 'cognacy'), ('cogid_cognateset_id', 'cogid'))

    #  TODO: re-read brown, implement identical procedure until graph.
    #   make it parametrizable whether we use asjp or other

    lex = lingpy.LexStat.from_cldf(ds.cldf_dir.joinpath('cldf-metadata.json'))
    args.log.info('Loaded the wordlist')

    G = nx.Graph()
    for idx, tokens, language in lex.iter_rows('tokens', 'doculect'):
        # args.log.info("tokens:"+str(tokens))
        for sound in iter_phonemes(tokens):
            if sound not in "+*-":
                # args.log.info("sound:"+ str(sound))
                sound = sound_class[sound]
                try:
                    G.node[sound]['frequency'][language] += 1
                except:
                    G.add_node(sound, frequency=Counter({language: 1}))

    # add the dummy node for gaps, in case we want to use it
    # G.add_node('-', frequency=Counter())

    # for node, data in G.nodes(data=True):
    #     data['language'] = ', '.join(sorted(data['language']))

    for lA, lB in progressbar(combinations(lex.cols, r=2)):
        for idxA, idxB in lex.pairs[lA, lB]:
            tokensA = list(iter_phonemes(lex[idxA, 'tokens']))
            tokensB = list(iter_phonemes(lex[idxB, 'tokens']))
            # args.log.info("---------")
            # args.log.info(tokensA)
            # args.log.info(tokensB)
            langA, langB = lex[idxA, 'doculect'], lex[idxB, 'doculect']
            # classesA, classesB = lex[idxA, 'classes'], lex[idxB, 'classes']

            # check for edit dist == 1giot
            potential_correspondence = lingpy.edit_dist(tokensA,
                                                        tokensB) <= args.threshold
            if potential_correspondence:
                pair = lingpy.Pairwise(tokensA, tokensB)
                pair.align()
                almA, almB, sim = pair.alignments[0]
                # args.log.info(almA)
                # args.log.info(almB)
                for soundA, soundB in zip(almA, almB):
                    if soundA != soundB and not '-' in [soundA,
                                                        soundB] and not '+' in [
                        soundA, soundB]:
                        soundA = sound_class[soundA]
                        soundB = sound_class[soundB]
                        try:
                            G[soundA][soundB]['frequency'][(langA, langB)] += 1
                        except:
                            G.add_edge(soundA, soundB,
                                       frequency=Counter({(langA, langB): 1}))

    # TODO: get genus level info
    # TODO: compute Correspondence percentage
    """Correspondence percentage is a more complex measure that corrects for 
    the frequency of individual sounds by recognizing the availability
    of a genus for a given correspondence. A genus is defined as ‘available’ for a corre-
    spondence between two sounds, X and Y, if it contains at least two languages of which
    one has two or more occurrences of X in the words making up its forty-item list and the
    other language has two or more occurrences of Y in its word list. X and Y can occur in
    any words, not necessarily words with the same meaning. The pairs of items are al-
    lowed to be different because availability should reflect the frequency of the sounds in-
    dependently of any relationship between them. The correspondence percentage is
    defined as the frequency of the correspondence divided by the total number of genera
    available for the correspondence, expressed as a percentage. In other words, the corre-
    spondence percentage is the percentage of available genera that actually show the cor-
    respondence.
    """
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
            freq_nA = sum(G.node[nA]['frequency'].values())
            freq_nB = sum(G.node[nB]['frequency'].values())
            G.node[nB]['total freq'] = freq_nB
            G.node[nA]['total freq'] = freq_nA
            data['num pairs'] = len(data['frequency'])
            data['total freq'] = freq_edge
    G.remove_edges_from(del_edges)

    table = []
    for nA, nB, data in sorted(G.edges(data=True), key=lambda x: x[2]['num pairs'],
                               reverse=True):
        data['language'] = ', '.join(
            ['{0}/{1}'.format(a, b) for a, b in sorted(data['frequency'])])

        if data['total freq'] >= args.cutoff:
            table += [[nA, G.node[nA]['total freq'],
                       nB, G.node[nB]['total freq'],
                       data['total freq'], data['num pairs']]]

    pos = nx.spring_layout(G)
    weights = [G[u][v]['num pairs'] for u, v in G.edges()]
    nx.draw(G,pos=pos, with_labels=True, width=weights)
    plt.show()

    with Table(
            args,
            *['Sound A', 'Freq A', 'Sound B', 'Freq B', 'Occ', 'Pairs'],
            rows=sorted(table, key=lambda x: (str(x[0]), str(x[2])))):
        pass
    #
    #
    # IG = networkx2igraph(G)
    # im = IG.community_infomap()
    # for i, comm in enumerate(im):
    #     for node in comm:
    #         IG.vs[node]['infomap'] = i+1
    #
    # IG.write_gml('{0}-graph.gml'.format(args.dataset))
