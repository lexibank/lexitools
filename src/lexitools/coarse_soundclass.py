#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coarsen sound classes"""
import pyclts
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import FrozenSet
import csvw

# Some CLTS sounds are composite
COMPOSITE = (pyclts.models.Diphthong, pyclts.models.Cluster)
CATEGORIES = {"consonant", "vowel", "cluster", "diphthong", "tone"}

@dataclass
class Rule():
    """ Abstract class for a coarsening rule """
    def apply(self, featureset):
        raise NotImplementedError()

@dataclass
class ChangeRule(Rule):
    """ Represents a change coarsening rule. """
    conditions: FrozenSet[tuple] = field(default_factory=frozenset)
    modification: FrozenSet[tuple] = field(default_factory=frozenset)

    def apply(self, featureset):
        if self.conditions <= featureset:
            # Remove everything in the condition
            for (f, v) in self.conditions:
                featureset.remove((f, v))
            for (f, v) in self.modification:
                # Remove any existing value for f
                for (f2, v2) in list(featureset):
                    if f2 == f:
                        featureset.remove((f2, v2))
                # Make the modification
                featureset.add((f, v))
@dataclass
class RemovalRule(Rule):
    """ Represents a removal coarsening rule.

    Attributes:
        feature: the feature to remove.
    """
    feature: str

    def apply(self, featureset):
        for (f, v) in list(featureset):
            if f == self.feature:
                featureset.remove((f, v))



class Coarsen(object):
    """Coarsens CLTS sounds by removing and replacing some features.

    This class coarsens sounds according to a configation which specifies which CLTS
    features should be replaced, and which should be removed.
    This allows the correspondence code to ignore noise linked to small variations in
    sound descriptions, at the cost of coarsening our view of sound correspondences.

    Coarsening maps sets of several CLTS sounds onto a single coarse sounds.
    The class carefully picks the most simple label for each set of sounds.

    For example, take a coarse sound with label "dz", defined by the coarsened features :
        `phonation=voiced category=consonant sibilancy=sibilant
         manner=affricate place=anterior`
    and which could result from coarsening the set of BIPA sounds:
        `{'dz', 'dzː', 'dz̪', 'dz̪ː', 'dzʰ', 'ˈʣʲ', 'ˈʣ',
         'ⁿdz', 'ⁿdzʱ', 'dzʱ', 'dzʲ', 'dzˤ'}`

    Attributes:
        bipa (pyclts.TranscriptionSystem): CLTS's BIPA system
        rules (dict of str to list of Rules): specification of the coarsening rules
        labels (dict): mapping of coarse feature frozensets to the corresponding coarse sound string.
        cache (dict): direct mapping of BIPA sound strings to coarse sound string.
    """

    cache = {}

    def __init__(self, bipa, config_path):
        """
        Coarsen sounds from BIPA according to a configuation.

        At initialization, we compute the corresponding coarse feature sets for all listed
         BIPA sounds, assign a label to each, and cache this information.
         Other BIPA sounds may be encountered after initialization,
         for which the coarse sound will be computed on the fly.


        The config file has the following shape, with 0 meaning a value deletion:

        ~~~
        TYPE,FEATURE,VALUE,ALTERED_FEATURE,ALTERED_VALUE,COMMENT
        vowel,relative_articulation,centralized,relative_articulation,0,
        vowel,relative_articulation,mid-centralized,relative_articulation,0,
        vowel,centrality,near-back,centrality,back,
        vowel,centrality,near-front,centrality,front,
        ~~~

        Example:
            >>> coarsen = Coarsen(clts.bipa, "default_coarsening.csv")
            >>> coarsen['dz̪']
            'dz'

        Args:
            bipa (pyclts.TranscriptionSystem): CLTS's BIPA system
            config_path (str): a path to a csv file for the config.

        """
        self.bipa = bipa
        self.rules = self._parse_config(config_path)
        self.labels = {}  # coarse feature set -> coarse string
        self.cache = {}  # bipa string -> coarse string

        # Construct a dict of coarse feature set -> all listed bipa sounds resulting in this set
        sounds = defaultdict(list)
        for sound in self.bipa.sounds:
            sound = self.bipa[sound]
            if isinstance(sound, COMPOSITE):
                coarse_f = self.get_coarse_features(sound.from_sound)
                sounds[coarse_f].append(sound.from_sound)
                coarse_f = self.get_coarse_features(sound.to_sound)
                sounds[coarse_f].append(sound.to_sound)
            else:
                coarse_f = self.get_coarse_features(sound)
                sounds[coarse_f].append(sound)

        # Populate coarsened and cache
        for f in sounds:
            name = self._create_label(f, sounds[f])
            self.labels[f] = str(name)
            for s in sounds[f]:
                self.cache[str(s)] = str(name)  # Instead, first try for the name

    def _parse_config(self, config_path):
        """ Parse a configuration table to generate coarse rules per category.

        Args:
            config_path: path to the config file.

        Returns:
            dict of categories to list of Rules
        """
        ANY = "#ANY#"
        def parse_features(feature_value_str):
            # ex: f=v&f2=v2
            if feature_value_str == "": ## empty set, used to remove features
                return
            for fv in feature_value_str.split("&"):
                f,v = fv.split("=")
                if v == "None":
                    v = None
                if v == "":
                    v = ANY
                yield f,v

        config = {}
        with csvw.UnicodeDictReader(config_path, delimiter=",") as reader:
            for row in reader:
                cat = row["TYPE"]
                if cat not in config:
                    config[cat] = []

                conditions = list(parse_features(row["CONDITIONS"]))
                # Removal rule
                if len(conditions) == 1 \
                    and conditions[0][1] == ANY \
                    and row["MODIFICATION"] == "":
                    rule = RemovalRule(feature=conditions[0][0])
                else:
                    modification = frozenset(parse_features(row["MODIFICATION"]))
                    rule = ChangeRule(conditions=frozenset(conditions),
                                  modification=modification)
                config[cat].append(rule)

        return config

    def _create_label(self, features, sound_set):
        """ Create a label for a coarse sound.

        A coarse sound is defined by a set of coarse features. Its label must be a
        BIPA sound which coarsens to this exact set of coarse features.
        In order to produce intuitive labels, we pick in bipa_sounds, preferring:

        - the shortest string
        - a sound which is as similar as possible to the coarse features
        - preferrably a string that is often repeated in the set of coarse sounds

        For example:
            >>> self.create_label(frozenset({("manner","approximant"),
            ...                   ("place","anterior"), ("phonation","voiced"),
            ...                   ("category","consonant")}), {"ɹ", "ɹ̩", "ð̞"})
            'ɹ'

        Args:
            features (frozenset): coarse feature-values which defines a coarse sound.
            sound_set (iterable of str): iterable of bipa strings which result in this coarse sound.

        Returns: a bipa sound which should serve as an alias for this feature set.
        """
        # hard codes the preference for a prototypical narrow place
        place_sort = {a:i for i,a in enumerate(["bilabial", "alveolar", "post-alveolar", "palatal"])}

        def jaccard(f1,f2):
            return len(f1 & f2) / len(f1 | f2)
        freqs = Counter([char for s in sound_set for char in str(s)])
        def cmp(sound):
            string_length = len(str(sound))
            try:
                place = sound.featuredict.get("place", None)
                this_featureset =  {item for item in sound.featuredict.items() if item[1] is not None}
                dist_to_coarse = -jaccard(features,this_featureset)
            except AttributeError: # Markers don't have featuredicts
                dist_to_coarse = 1
                place = None
            negfreq = -sum(freqs[c] for c in str(sound))

            place_order = place_sort.get(place, len(place_sort)+1)
            return (string_length, place_order, dist_to_coarse, negfreq)

        return min(sound_set, key=cmp)

    def __getitem__(self, item):
        """ Get a coarse sound string from a BIPA sound string.

        If the sound is known, retrieve it form the cache, otherwise, generate and cache
        its coarse equivalent.

        Special care must be given to clts diphthongs and clusters, which are represented
        as "composite" sounds, made of two simple distinct sounds, `sound.from_sound`
        and `sound.to_sound`. When encountering these, we coarsen each separately, then
        assemble them. This may result in twice the same sound, if their difference was
        levelled by the coarsening (e.g. "ɪi" -> "ii"), which bipa would read as a long sound
        ("iː"). In this case, we again coarsen the resulting sound, as length might also
        be levelled by the coarsening (resulting in "i").

        Args:
            item (str): a BIPA sound string.

        Returns:
            coarse (str): the corresponding coarse sound label.
        """
        try:
            return self.cache[item]
        except KeyError:
            sound = self.bipa[item]
            if isinstance(sound, pyclts.models.UnknownSound):
                # Unknown CLTS sounds get their own coarse class
                coarse_sound =  item
                fs = frozenset({("unknownSound", str(item))})
                self.labels[fs] = item
                #raise ValueError("Unknown sound " + item)
            elif isinstance(sound, COMPOSITE):
                sa = self.coarsen_sound(sound.from_sound)
                sb = self.coarsen_sound(sound.to_sound)
                new_sound = self.bipa[sa + sb]
                if not isinstance(new_sound, COMPOSITE + (pyclts.models.UnknownSound,)):
                    # we reduced this into a simple sound, needs further coarsening
                    coarse_sound = self[str(new_sound)]
                else:
                    coarse_sound = sa + sb
            else:
                coarse_sound = self.coarsen_sound(sound)
            self.cache[item] = coarse_sound
            return coarse_sound

    def coarsen_sound(self, simple_sound):
        """ Coarsen a yet unknown BIPA sound.

        The sound can *not* be composite, such as diphthongs and clusters.
        Even though the BIPA sound is yet unknown, the corresponding coarse sound might
        already be known, in which case we can get its label from `self.labels`. If the
        coarse sound is entirely unknown, we generate a new label.

        Args:
            simple_sound (pyclts.Sound): a BIPA sound

        Returns:
            coarse (str): a coarse sound label
        """
        f = self.get_coarse_features(simple_sound)
        try:
            return self.labels[f]
        except KeyError:
            self.labels[f] = str(self._create_label(f, {simple_sound}))
            return str(simple_sound)

    def get_coarse_features(self, sound):
        """ Get Coarse features from a BIPA  sound.

        Args:
            sound (pyclts.Sound): BIPA sound.

        Returns:
            coarse (frozenset): Coarse features-value pairs defining a coarse sound.

        """
        cat = sound.type
        try:
            features = set(sound.featuredict.items())
        except AttributeError:
            # markers do not have featuredicts in CLTS
            features = {("marker", str(sound))}
        for rule in self.rules.get(cat,[]):
            rule.apply(features)
        features.add(("category", cat))
        features = frozenset({(f,v) for f,v in features if v is not None})
        return features

    def as_table(self):
        """ Describe all known sounds as a table, ready for csv export.

        Creates a table as a list of rows. The table includes a header.
        Each row is alist of strings and represents a coarse sound.
        The columns are:
            - "BIPA": the list of known BIPA sounds which result in this coarse sound.
            - "Coarse": the label of this coarse sound
            - "Coarse features": the features and values which define this coarse sound
                (features and values are separated by "=", as in "height=close").

        Returns:
            rows (list of list): list of known coarse sounds and their BIPA counterparts.
        """
        reversed_cache = defaultdict(list)
        for bipa in self.cache:
            coarse = self.cache[bipa]
            reversed_cache[coarse].append(bipa)
        rows = [["BIPA", "Coarse", "Coarse features"]]
        for fs in self.labels:
            coarse = self.labels[fs]
            all_bipa = reversed_cache[coarse]
            rows.append(
                [" ".join(all_bipa), coarse, " ".join(sorted(f + "=" + v for f, v in fs))])
        return rows
