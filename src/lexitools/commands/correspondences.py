"""
Extract correspondences from a set of datasets.
"""
from clldutils.clilib import add_format
from cldfbench.cli_util import add_catalog_spec
from pyconcepticon.api import Concepticon
from cldfcatalog import Config
import lingpy
import lingpy.evaluate
from lingpy.compare import partial
from lingpy.compare.util import mutual_coverage_check
from lingpy.compare.sanity import average_coverage
from collections import Counter, defaultdict, namedtuple
from pylexibank import progressbar
import pyglottolog
import csv
import json
from lexitools.coarse_soundclass import Coarsen
import time
import logging
import lingrex
from pathlib import Path
from cldfbench_lexibank_analysed import Dataset
from multiprocessing import Pool

# Do not merge consecutive vowels into diphthongs
lingpy.settings.rcParams["merge_vowels"] = False

# import cProfile
# from memory_profiler import profile


# set of characters which should be ignored in tokens (markers, etc)
IGNORE = {"+", "*", "#", "_", ""}

"""A Language has a name, a glottocode, and a family."""
Lang = namedtuple("Lang", ('glottocode', 'family', 'name'))


class NotEnoughDataError(ValueError):
    pass


class MagicGap(str):
    """ A gap with context, which Lingrex recognizes as a gap.

        In Lingrex: `x == '-'` is used to check if a token is a gap.

        We need symbols that pass that test, but which carry contextual information,
            and are different from Magicgaps in different contexts.
    """

    def __new__(cls, left, right, *args, **kwargs):
        s = str.__new__(cls, "-")
        s.left = left
        s.right = right
        return s

    def __ne__(self, other): return not self == other

    def __eq__(self, other):
        if type(other) is MagicGap: return str(self) == str(other)
        return other == "-"

    def __str__(self): return self.left + "|-|" + self.right

    def __hash__(self): return hash("-")


class LexicoreData(object):
    """ Load and prepare lexicore data.

    Attributes:
        errors (list of list): table which summarizes errors encountered in tokens.
        concepts_subset (set): the set of all concepts which will be kept.
        lang_to_concept (defaultdict): mapping of Lang to concepts.
        _data (defaultdict): mapping of (lang, concept) -> token string -> word instances
           A language can have more that one token for a specific
         concept, and a single token can be found in more than one dataset. A Word represents
         exactly one token in a given language and dataset.
    """

    def __init__(self, coarsen, glottolog, subset=None, concept_list=None):
        """ Initialize the data by collecting, processing and organizing tokens.

        Iterates over the database, to:
        - Identify the family for all languages
        - If specified, restrict data to a specific list of concepts.
        - Pre-process tokens to replace sounds by sound classes and count syllables
        - Make note of any tokens which result in errors.

        Data is separated by family and prepared as a lingpy WordList internal dictionnary,
        so that it is ready for loading in lingpy.

        Args:
            concept_list (str): name of a concepticon concept list
            coarsen (Coarsen): a coarsening instance for sounds
            glottolog (pyglottolog.Glottolog): Glottolog instance to retrieve family names.
        """
        datasets = [(ds.properties['rdf:ID'], ds)
                    for ds in Dataset()._datasets(set_="LexiCore")]
        if subset:
            datasets = list(filter(lambda x: x[0] in subset, datasets))

        self.datasets_ids = [id for id, _ in datasets]

        # This mixes some lingpy specific headers (used to init lexstat)
        # and some things we need specifically for this application
        namespace = ['doculect', 'concept', 'concepticon', 'original_token',
                     'original_id', 'tokens', 'structure',
                     'glottocode', 'cogid', 'cldf_dataset']

        self.cols = {n: i for i, n in enumerate(namespace)}
        self.stats = {}
        self.coarsen = coarsen

        # Obtain glottocode and family:
        # family can be obtained from glottolog (this is slow) if glottolog is loaded
        # if there is no glottocode, we can still use the hardcoded family
        self.languages = self._prepare_languages(datasets, glottolog)

        self.errors = [["Dataset", "Language_ID", "Sound", "Token", "ID"]]

        concepts = {row["ID"]: (row["Concepticon_ID"], row["Concepticon_Gloss"])
                    for name, ds in datasets for row in ds['ParameterTable']}

        # Define concept list
        if concept_list is not None:
            concepticon = Concepticon(Config.from_file().get_clone("concepticon"))
            concept_dict = concepticon.conceptlists[concept_list].concepts
            self.concepts_subset = {c.concepticon_id for c in
                                    concept_dict.values() if
                                    c.concepticon_id}
        else:  # All observed concepts
            self.concepts_subset = {i for i, j in concepts.values()}

        # dict of family -> (language, concept, word) -> row
        self._data = defaultdict(lambda: [namespace])
        self.cogids = defaultdict(set)

        duplicates = Counter()

        for dataset_id, ds in progressbar(datasets,
                                          desc="Loading data..."):
            for row in ds['FormTable']:
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


                doculect = row["Language_ID"]
                lang = self.languages[doculect]


                # Convert sounds, ignore rows with unknown or ignored sounds
                try:
                    tokens = list(self._iter_phonemes(row))
                    structure = [t if t in IGNORE else "A" for t in tokens]
                except ValueError:
                    continue  # unknown sounds
                if all([s in IGNORE for s in tokens]): continue

                cogid = row.get("Cognacy", row.get("Cognateset_ID", None))


                self.cogids[(lang.family, dataset_id)].add(cogid)

                # skip any duplicates.
                ipa_word = tuple(tokens)
                if duplicates[(doculect, concept_id, ipa_word)] > 0: continue

                duplicates[(doculect, concept_id, ipa_word)] += 1

                # add row
                self._data[lang.family].append([doculect, concept_gloss, concept_id,
                                                " ".join(row["Segments"]), row["ID"],
                                                tokens, structure,
                                                lang.glottocode,
                                                cogid, dataset_id])

        self._prepare_cogids()
        self.families = list(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def output(self, output_prefix):
        filenames = []
        for family in self.families:
            filename = f'{output_prefix}_data_{family}.csv'
            filenames.append(filename)
            with open(filename, 'w',
                      encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=',', )
                writer.writerows(self._data[family])

    def iter_table(self, table_name):
        """ Iter on the rows of a specific table, across all loaded datasets.

        Args:
            table_name (str): name of a CLDF Wordlist table.

        Yields:
            rows from all datasets, with an additional "dataset" field indicating the name
            of the dataset from which the row originates.
        """
        for name, ds in self.datasets:
            for row in ds[table_name]:
                row["dataset"] = name
                yield row

    def _prepare_cogids(self):
        """ Prepare cogids for analysis by turning cognate ids to ints (or lists of ints).

        We need all cogids to be ints, or in case of partial cogids, lists of ints.
        However, often they are strings.

        Note that cognate IDs are "local" to datasets, not global to LexiCore.
        """
        cog_to_int = {}
        next_cogid = 1

        def single_cogid(dataset, cogid):
            nonlocal next_cogid
            if (dataset, cogid) not in cog_to_int:
                cog_to_int[(dataset, cogid)] = next_cogid
                next_cogid += 1
            return cog_to_int[(dataset, cogid)]

        def multiple_cogids(dataset, cogids):
            return tuple(single_cogid(dataset, x) for x in cogids)

        # For each dataset, determine whether cogids are lists or ints
        partial = {}
        for family, dataset in self.cogids:
            partial[dataset] = any(
                c is not None and " " in c for c in self.cogids[(family, dataset)])

        # Convert cogids to either ints or lists of ints
        c = self.cols["cogid"]
        t = self.cols["original_token"]
        d = self.cols["cldf_dataset"]
        for family in self._data:
            for i, row in enumerate(self._data[family]):
                if i > 0:
                    cogid = row[c]
                    dataset = row[d]
                    if cogid in {"0", 0, None, ""}:
                        row[c] = None
                    elif partial[dataset]:
                        cogids = cogid.split(" ")
                        morphemes = row[t].split(" + ")
                        if len(cogids) != len(morphemes):
                            # Sometimes, I find a single cogid with a word which has multiple morphemes
                            # I think this happens because of words which have multiple
                            # morphemes but have no cognates in the dataset.
                            row[c] = tuple(single_cogid(dataset, cogid) for _ in
                                           range(len(morphemes)))
                        else:
                            row[c] = multiple_cogids(dataset, cogids)
                    else:
                        row[c] = single_cogid(dataset, cogid)

    def _prepare_languages(self, datasets, glottolog):
        """ Prepare a list of all languages in the data.

        For each language, we collect its family, glottocode & local name in the dataset.

        This mostly uses glottocode to grab the family name, and corrects isolate family
        names to avoid having a single "Isolate" family with all isolates !

        Args:
            glottolog (pyglottolog.Glottolog): Glottolog instance to retrieve family names.

        Returns:
            languages: a dictionnary of language ID in datasets to a Lang instance.
        """
        # this is less slow than calling .langoid(code) for each code
        langoids = glottolog.languoids_by_code()
        languages = {}
        for name, ds in progressbar(datasets, desc="listing languages"):
            for row in ds["LanguageTable"]:
                gcode = row["Glottocode"]
                if gcode is None: continue
                family = row["Family"]
                langoid = langoids.get(gcode, None)
                if langoid is not None and langoid.family is not None:
                    family = langoid.family.name
                if family == "Isolate":
                    family = langoid.name + "_isolate"

                languages[row["ID"]] = Lang(family=family, glottocode=gcode, name=row[
                    "Name"])  # TODO: should we use the langoid name here ?
        return languages

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
                if (
                        i == 0 or i == l - 1) and segment == "+":  # ignore initial and final "+"
                    continue
                yield segment
            except ValueError as e:
                self.errors.append((row["dataset"], row["Language_ID"], segment,
                                    " ".join(str(x) for x in segments), row["ID"]))
                raise e


class CorrespFinder(object):

    def __init__(self, data, family, prefix, languages, categories):
        """ Class for finding sound correspondences in a given family. """
        data = dict(enumerate(data))
        cols = {n: i for i, n in enumerate(data[0])}
        self.categories = categories

        logging.info(f"Preparing data for {family}")
        self.languages = languages
        self.prefix = prefix
        cogids_counts = Counter()
        self.segmented = False
        self.gold_segmented = True

        for i, row in data.items():
            cogid = row[cols["cogid"]]
            if "+" in row[cols["tokens"]]:
                self.segmented = True
            if type(cogid) is not tuple:
                self.gold_segmented = False
            cogids_counts[cogid] += 1
        self.needs_prediction = None in cogids_counts
        self.partial = (self.needs_prediction and self.segmented) \
                       or (self.gold_segmented)

        # Create wordlist instances for cognate detection
        if self.segmented:  # Cognates and alignments are morpheme-level
            self.lex = partial.Partial(data, check=True)
        else:  # Cognates and alignments are word-level
            self.lex = lingpy.LexStat(data, check=True)

        # Check if this is usable
        if self.lex.height < 2 or self.lex.width < 2:
            datasets = ", ".join(
                {r[cols["cldf_dataset"]] for i, r in data.items() if i > 0})
            raise NotEnoughDataError(f"Family {family} has no usable data "
                                     f"(datasets: {datasets})")

        self.infos = sanity_stats(self.lex)
        self.infos["evals"] = {}
        self.infos["family"] = self.family = family


    def find_correspondences(self, eval=False):
        """ Iterator over correspondence patterns.

        Finds correspondence patterns:

        - If needed, find cognates automatically and/or evaluate cognate detection
            (lingpy)
        - Align cognate sets (lingpy)
        - Replace sounds by sounds with contexts in the alignments
        - Consolidate alignment sites into patterns (lingrex)
        - Iterate over correspondence patterns, and yield rows for the output table.
            One row is a single sound of a correspondence pattern.

        Args:
            eval (bool): whether to evaluate cognate detection

        Returns:

        """

        def format_ex(alm, i):
            template = "{word} ([{original_token}] {cldf_dataset} {original_id})"
            return template.format(word=" ".join(alm[i, "tokens"]),
                                   original_token=alm[i, "original_token"],
                                   cldf_dataset=alm[i, "cldf_dataset"],
                                   original_id=alm[i, "original_id"])

        lingpy.log.get_logger().setLevel(logging.WARNING)

        # Find cognates
        logging.info(f"Searching for cognates in {self.family}")
        self._find_cognates(eval)

        # Align cognates
        logging.info(f"Aligning cognates in {self.family}")
        self.lex = lingpy.Alignments(self.lex, ref="pred_cogid", fuzzy=self.partial)
        self.lex.align(method='library', iteration=True, model="sca", mode="global")

        # Override alignment to add contexts (no loss of info)
        logging.info(f"Adding contexts in {self.family}")
        msa = self.lex.msa["pred_cogid"]
        for cogid in msa:
            msa[cogid]["alignment"] = [list(self.add_contexts(seq)) for seq in
                                       msa[cogid]["alignment"]]

        # Consolidate alignment sites into patterns
        logging.info(f"Clustering alignment sites into patterns in {self.family}")
        self.lex = lingrex.CoPaR(self.lex, ref="pred_cogid", fuzzy=self.partial,
                                 segments="tokens", structure="structure")
        self.lex.get_sites()
        self.lex.cluster_sites()

        langs = self.lex.cols

        logging.info(f"Preparing corresp patterns in {self.family}")
        # Iterate over patterns
        for pat_idx, ((_, pattern), sites) in enumerate(self.lex.clusters.items()):
            # Unique identifier for this pattern
            pattern_id = self.family + "-" + str(pat_idx)
            # print(pattern)

            site_count = len(sites)  # Number of sites displaying the pattern
            sites_refs = [dict(zip(self.lex.msa["pred_cogid"][cog]["taxa"],
                                   self.lex.msa["pred_cogid"][cog]["ID"])) for cog, pos in
                          sites]
            # print([cp.msa["pred_cogid"][cog] for cog, pos in sites])

            # yield one row for each sound in the pattern
            for i in range(len(pattern)):
                lang = langs[i]
                sound = str(pattern[
                                i])  # if it's a magicGap, this converts it to a context string

                if sound == 'Ø' or sound in IGNORE: continue

                # Format examples
                examples = []
                for ref in sites_refs:
                    if lang in ref:  # If this alignment had this language
                        examples.append(format_ex(self.lex, ref[lang]))
                    if len(examples) == 3: break

                yield [self.family, pattern_id, site_count, *sound.split("|"),
                       self.languages[lang].name,
                       self.languages[lang].glottocode,
                       ";".join(examples)]
        # TODO: notes for Erich: having this context specific makes it a little sparser, might want to merge them.

    def _find_cognates(self, eval):
        """ Perform cognate detection & evaluation in a given family.

        - Detect cognates in families
            - if evaluation must be run, then also detect  cognates in families where we have gold data for everything.
         - Record information about the process
         - Run evaluation if needed

        Since we run the prediction on a per-family basis, and there can be multiple
        datasets with data from the same family, we will use predicted cognates for the
        entire family if even a single dataset doesn't have gold data. If we were instead
        predicting only for the datasets which don't have gold data, we would need to match
        these predicted labels to the gold labels, and it is not clear how to do that,
        or whether this would be better than just predicting for everything.

        The evaluation itself will be performed on a per-dataset basis.

        Args:
            family (str): family name
            pbar: progressbar
            eval (bool): whether to evaluate cognate detection
            pred_params (dict): informations regarding whether the data has gold cognates,
             and whether the cognates IDs are morpheme based or word based

        Returns:

        """
        # prediction needed when we don't have all cogids, or when evaluating
        if self.needs_prediction or eval:
            # Decide which algorithm to run
            kw = dict(method='lexstat', threshold=0.55, ref="pred_cogid",
                      cluster_method='infomap')

            if self.infos["min_mutual_coverage"] < 100:
                kw = dict(method="sca", threshold=0.45, ref='pred_cogid')

            # Run cognate detection
            if self.segmented:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.infos["detection_type"] = 'morpheme'
                self.infos["cognate_source"] = "Partial(" + kw_str + ")"
                self.lex.get_scorer(runs=1000)
                self.lex.partial_cluster(**kw)
            else:
                kw_str = ", ".join("=".join([k, str(v)]) for k, v in kw.items())
                self.infos["detection_type"] = 'word'
                self.infos["cognate_source"] = "LexStat(" + kw_str + ")"
                self.lex.get_scorer(runs=1000)
                self.lex.cluster(**kw)

        ## Evaluate cognate detection
        if eval:
            self._eval_per_dataset()

        # Either we didn't predict, or we predicted only for evaluation purposes
        if not self.needs_prediction:
            self.infos["detection_type"] = 'morpheme' if self.gold_segmented else 'word'
            self.lex.add_entries("pred_cogid", "cogid", lambda x: x, override=True)
            self.infos["cognate_source"] = "expert"

    def _eval_per_dataset(self):
        """ Evaluate cognate detection for each dataset in a family.

        - Split the WordList instance per dataset
        - Iterate datasets, and evaluate the prediction (either partial or whole word)
        - update the self.evals dictionnary


        Args:
            lex (WordList): this is either a Partial or a Lexstat instance
            family (str): family name
            pred_params (dict): informations regarding whether the data has gold cognates,
             and whether the cognates IDs are morpheme based or word based
        """
        eval_measures = ["precision", "recall", "f-score"]
        # separate data by dataset, keep only if gold annotation exists
        columns = self.lex.columns
        by_datasets = defaultdict(lambda: [list(columns)])
        for i, r in self.lex._data.items():
            if i > 0 and r[self.lex._header['cogid']] is not None:
                dataset = r[self.lex._header['cldf_dataset']]
                by_datasets[dataset].append(
                    list(r))  # copy of rows, otherwise we edit lex

        # Evaluate inside each dataset
        for dataset in progressbar(list(by_datasets), desc="Evaluating each dataset..."):
            gold_rows = dict(enumerate(by_datasets[dataset]))

            eval_lex = lingpy.Wordlist(gold_rows)

            if eval_lex.height < 2 or eval_lex.width < 2:
                logging.warning(f"Dataset {dataset} in family {self.family}"
                                f" has no usable eval data")
                continue

            if self.segmented and self.gold_segmented:
                try:
                    res = lingpy.evaluate.acd.partial_bcubes(eval_lex, gold='cogid',
                                                         test='pred_cogid',
                                                         pprint=False)
                except TypeError:
                    print("\n\n\n")
                    print(dataset)
                    print("\n\n\n")
                    raise


                # Diff won't work with lists, needs a tuple
                eval_lex.add_entries("pred_cogid", "pred_cogid",
                                     tuple,
                                     override=True)
            else:
                if self.segmented:  # revert to word-level cognates
                    eval_lex.add_entries("pred_cogid", "pred_cogid",
                                         lambda x: " ".join(str(i) for i in x),
                                         override=True)

                res = lingpy.evaluate.acd.bcubes(eval_lex, gold='cogid',
                                                 test='pred_cogid',
                                                 pprint=False)
            lingpy.evaluate.acd.diff(eval_lex, 'cogid', 'pred_cogid', tofile=True,
                                     filename=f"{self.prefix}_"
                                              f"{self.family}_"
                                              f"{dataset}_cognate_eval",
                                     pprint=False)

            d = dict(zip(eval_measures, res))
            d.update(sanity_stats(eval_lex))
            d["cognate_source"] = self.infos["cognate_source"]
            d["cognates"] = len({cog for i, cog in eval_lex.iter_rows("cogid")})
            d["languages"] = eval_lex.width
            d["tokens"] = len(eval_lex)
            d["detection_type"] = "morpheme" if self.segmented else "word"
            self.infos["evals"][dataset] = d
            del by_datasets[dataset]

    def add_contexts(self, seq):
        """ Iterator of sounds and contexts for a pair of aligned tokens.

        Args:
            seq (list of str): sequence of segments or gap.

        Yields: pair of aligned sounds and their contexts: `(sound, context)`
        """

        def to_categories(sequence):
            """Turn a sequence of sounds into a sequence of categories used in contexts"""
            for s in sequence:
                if s == "-":
                    yield None
                elif s in IGNORE:
                    yield s
                else:
                    cat = self.categories[s]
                    if cat in {"T"}:
                        yield None
                    else:
                        yield cat
            yield "#"

        def get_right_context(cats, i):
            """Return the context for a given sound."""
            return next((c for c in cats[i + 1:] if c is not None))

        cats = list(to_categories(seq))
        l = len(seq)
        left = "#"
        for i in range(l):
            right = get_right_context(cats, i)
            if seq[i] == "-":  # output a gap which retains context info
                yield MagicGap(left, right)
            elif seq[i] in IGNORE:  # output markers as is
                yield seq[i]
            else:
                yield left + "|" + seq[i] + "|" + right
            left = left if cats[i] is None else cats[i]


def sanity_stats(lexicon):
    """Useful sanity stats for cognate recognition.

    Uses standard lingpy functions.
    """

    def mut_cov():
        for i in range(lexicon.height, 1, -1):
            if mutual_coverage_check(lexicon, i):
                return i
        return 0

    d = {"min_mutual_coverage": mut_cov()}

    try:
        d["average_coverage"] = average_coverage(lexicon)
    except ZeroDivisionError:
        d["average_coverage"] = 0
    return d

def register(parser):
    # Standard catalogs can be "requested" as follows:
    add_catalog_spec(parser, "clts")
    add_catalog_spec(parser, "glottolog")
    add_format(parser, default='pipe')

    parser.description = run.__doc__

    parser.add_argument(
        '--cpus',
        default=1,
        type=int,
        help='Number of cpus to use for multithreading'
    )
    parser.add_argument(
        '--subset',
        default=None,
        nargs='+',
        help='Space-separated list of dataset IDs. '
             'Select only a few lexicore datasets (otherwise entire lexicore)'
    )

    parser.add_argument(
        '--cognate_eval',
        action='store_true',
        default=False,
        help='Evaluate cognate detection.')

    parser.add_argument(
        '--output',
        type=Path,
        default=Path("results/"),
        help='output folder')

    parser.add_argument(
        '--concepts',
        action='store',
        default=None,
        type=str,
        help='select a concept list to filter on')


def run(args):
    """Run the correspondence command.

    Run with:

        cldfbench lexitools.correspondences --clts-version v1.4.1

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
    args.output.mkdir(exist_ok=True, parents=True)

    coarsening_file = (
            Path(__file__) / "../../../../etc/default_coarsening.csv").resolve()
    coarse = Coarsen(clts.bipa, str(coarsening_file))

    output_prefix = str(args.output / f"{now}_sound_correspondences")

    data = LexicoreData(coarse,
                        pyglottolog.Glottolog(args.glottolog.dir),
                        concept_list=args.concepts,
                        subset=args.subset)

    # data.output(output_prefix) # In case we want intermediary output

    args.log.info(
        f'Loaded the wordlist ({len(data.languages)} languages, '
        f'{len(data.families)} families, '
        f'{len(data.concepts_subset)} concepts kept)')

    with open(f'{output_prefix}_coarsening.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerows(coarse.as_table())

    with open(output_prefix + '_metadata.json', 'w',
              encoding="utf-8") as metafile:
        json.dump({"concepts": args.concepts,
                   "dataset_list": sorted(data.datasets_ids),
                   "n_families": data.families,
                   "n_concepts": len(data.concepts_subset),
                   "n_tokens": sum([len(data[f]) - 1 for f in data.families])},
                  metafile, indent=4, sort_keys=True)

    with open(output_prefix + '_sound_errors.csv', 'w',
              encoding="utf-8") as errorfile:
        for line in data.errors:
            errorfile.write(",".join(line) + "\n")

    # Multithread search for corresp patterns

    if args.cpus > 1:
        pool = Pool(args.cpus)
        map = pool.imap_unordered

    infos = progressbar(map(process_family,
                            ((data[family], family, output_prefix, data.languages,
                              coarse.categories, args.cognate_eval) for family
                             in data.families)
                            ), total=len(data.families))

    export_infos = []
    for info in infos:
        export_infos.append(info)
        logging.info(f"Finished running for family {info['family']}")
        # attempt to free up some memory...
        del data[info['family']]

    with open(output_prefix + f'_extraction_infos.json', 'w',
              encoding="utf-8") as infos_file:
        json.dump(export_infos, infos_file, indent=4, sort_keys=True)

def process_family(args):
    """ Find correspondences for a single family and write results.

    This can be run in parallel on each family.

    Args:
        args: Tuple with all arguments for CorrespFinder

    Returns:
        infos (dict): information on cognate inference & evaluation.
    """
    data, family, output_prefix, languages, categories, eval = args
    try:
        corresp_finder = CorrespFinder(data, family, output_prefix, languages, categories)
    except NotEnoughDataError:
        return None

    # Find all correspondences and write
    with open(f'{output_prefix}_{family}_counts.csv', 'w',
              encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(
            ["family", "pat_id", "count", "left_context", "sound", "right_context",
             "language", "glottocode", "examples"])
        writer.writerows(corresp_finder.find_correspondences(eval))

    return corresp_finder.infos


# TODO: memory management with multithreading
# TODO: In the same family (e.g. Sino-Tibetan), some datasets might have eval data which is segmented,
# and some might have eval data which is not segmented.
# We do the eval per dataset, but should we re-predict for each eval dataset using the same param for segmentation ?