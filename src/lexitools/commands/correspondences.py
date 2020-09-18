"""
Check the prosodic structure of a given dataset.
"""

from clldutils.clilib import Table, add_format
from cldfbench.cli_util import add_catalog_spec, get_dataset

from linse.transform import syllable_inventories
from pylexibank.cli_util import add_dataset_spec

# lingpy for alignments
from lingpy import *
# need this to store as GML format (for inspection)
from lingpy.convert.graph import networkx2igraph

# needed for cross-check
from itertools import combinations

from pylexibank import progressbar

# networkx
import networkx as nx



def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")

    # Require a dataset as argument for the command:
    add_dataset_spec(parser)
    add_format(parser, default='pipe')

    #parser.add_argument(
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


def run(args):

    ds = get_dataset(args)
    clts = args.clts.from_config().api
    bipa = clts.transcriptionsystem_dict['bipa']
    color = clts.soundclasses_dict['color']
    asjp = clts.soundclasses_dict['asjp']
    sca = clts.soundclasses_dict['sca']


    columns=('parameter_id', 'concept_name', 'language_id', 'language_name',
            'value', 'form', 'segments', 'language_glottocode',
            'concept_concepticon_id', 'language_latitude',
            'language_longitude')
    namespace=(('concept_name', 'concept'), ('language_id', 'doculect'),
            ('segments', 'tokens'), ('language_glottocode', 'glottolog'),
            ('concept_concepticon_id', 'concepticon'), ('language_latitude',
                'latitude'), ('language_longitude', 'longitude'), ('cognacy',
                    'cognacy'), ('cogid_cognateset_id', 'cogid'))


    lex = LexStat.from_cldf(ds.cldf_dir.joinpath('cldf-metadata.json'))
    args.log.info('Loaded the wordlist')

    G = nx.Graph()
    for idx, tokens, language in lex.iter_rows('tokens', 'doculect'):
        for sound in [t.split('/')[1] if '/' in t else t for t in tokens]:
            try:
                #G.node[sound]['frequency'] += 1
                G.node[sound]['language'].add(language)
            except:
                G.add_node(
                        sound, 
                        frequency=1, 
                        language=set([language]),
                        name=bipa[sound].name, 
                        color=color[sound],
                        asjp=asjp[sound],
                        sca=sca[sound],
                        type=bipa[sound].type
                        )
                
    # add the dummy node for gaps, in case we want to use it
    G.add_node('-',frequency=0, language=set(lex.cols))

    for node, data in G.nodes(data=True):
        data['language'] = ', '.join(sorted(data['language']))


    for lA, lB in progressbar(combinations(lex.cols, r=2)):
        for idxA, idxB in lex.pairs[lA, lB]:
            tokensA, tokensB = (
                    [t.split('/')[1] if '/' in t else t for t in lex[idxA, 'tokens']],
                    [t.split('/')[1] if '/' in t else t for t in lex[idxB, 'tokens']]
                    )
            langA, langB = lex[idxA, 'doculect'], lex[idxB, 'doculect']
            classesA, classesB = lex[idxA, 'classes'], lex[idxB, 'classes']

            # check for edit dist == 1
            if edit_dist(
                    classesA, 
                    classesB
                        ) <= args.threshold:
                pair = Pairwise(tokensA, tokensB)
                pair.align()
                almA, almB, sim = pair.alignments[0]
                for soundA, soundB in zip(almA, almB):
                    if soundA != soundB and not '-' in [soundA, soundB] and not '+' in [soundA, soundB]:
                        G.node[soundA]['frequency'] += 1
                        G.node[soundB]['frequency'] += 1
                        try:
                            G[soundA][soundB]['frequency'] += 1
                            G[soundA][soundB]['language'].add((langA, langB))
                        except:
                            G.add_edge(
                                    soundA, 
                                    soundB, 
                                    frequency=1, 
                                    language=set([(langA, langB)])
                                    )

    
    # add weights and delete, using a normalization formula (from Dellert)
    delis = []
    for nA, nB, data in G.edges(data=True):
        data['weight'] = data['frequency']**2/(
                G.node[nA]['frequency']+G.node[nB]['frequency']-data['frequency'])
        if data['frequency'] < args.cutoff:
            delis += [(nA, nB)]
    G.remove_edges_from(delis)

    table = []
    for nA, nB, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True):

        data['language'] = ', '.join(['{0}/{1}'.format(a, b) for a, b in sorted(
            data['language'])])
        

        if data['frequency'] >= args.cutoff:
            table += [[nA, 
                G.node[nA]['frequency'], 
                nB, 
                G.node[nB]['frequency'], 
                data['frequency'],
                data['weight'],
                G.node[nA]['type']]]

    with Table(
            args, 
            *['Lang A', 'Freq A', 'Lang B', 'Freq B', 'Occ', 'Weight'], 
            rows=sorted(table, key=lambda x: (x[-1], x[-2]))):
        pass


    IG = networkx2igraph(G)
    im = IG.community_infomap()
    for i, comm in enumerate(im):
        for node in comm:
            IG.vs[node]['infomap'] = i+1

    IG.write_gml('{0}-graph.gml'.format(args.dataset))

