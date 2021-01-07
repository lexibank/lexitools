#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coarsen sound classes"""
import pyclts
from collections import defaultdict

# Some CLTS sounds are composite
COMPOSITE = (pyclts.models.Diphthong, pyclts.models.Cluster)

# Default configuration, used by the correspondence command in lexitools.
DEFAULT_CONFIG = dict(
    remove={'advanced', 'advanced-tongue-root', 'apical', 'aspirated', 'breathy',
            'centralized', 'creaky', 'devoiced', 'ejective', 'glottalized', 'labialized',
            'labio-palatalized', 'laminal', 'less-rounded', 'long', 'lowered',
            'mid-centralized', 'mid-long', 'more-rounded',
            # 'nasalized',
            'non-syllabic',
            'palatalized', 'pharyngealized', 'pre-aspirated', 'pre-glottalized',
            'pre-labialized', 'pre-nasalized', 'pre-palatalized', 'primary-stress',
            'raised', 'retracted', 'retracted-tongue-root','revoiced', 'rhotacized',
            'secondary-stress', 'strong', 'syllabic', 'ultra-long', 'ultra-short',
            'unreleased', 'velarized', 'with-frication', 'with-lateral-release',
            'with-mid-central-vowel-release', 'with-nasal-release', 'with_downstep',
            'with_extra-high_tone', 'with_extra-low_tone', 'with_falling_tone',
            'with_global_fall', 'with_global_rise', 'with_high_tone', 'with_low_tone',
            'with_mid_tone', 'with_rising_tone', 'with_upstep',
             },
    change={'alveolar': 'anterior', 'alveolo-palatal': 'palatal', 'close-mid': 'mid',
          'dental': 'anterior', 'linguolabial': 'labial', 'nasal-click': 'click',
          'near-back': 'back', 'near-close': 'close', 'near-front': 'front',
          'near-open': 'open',
          'open-mid': 'mid', 'palatal-velar': 'velar',
          'post-alveolar': 'palatal', 'tap': 'vibrant', 'trill': 'vibrant'})

class Coarsen(object):
    """Coarsens CLTS sounds by removing and replacing some features.

    This class coarsens sounds according to a configation which specifies which CLTS
    features should be replaced, and which should be removed.
    This allows the correspondence code to ignore noise linked to small variations in
    sound descriptions, at the cost of coarsening our view of sound correspondences.

    Coarsening maps sets of several CLTS sounds onto a single coarse sounds.
    The class carefully picks the most simple label for each set of sounds.

    For example, the default configuration given in this file creates a coarse sound "dz",
    with coarsened features "sibilant anterior voiced affricate consonant",
    and which result  from coarsening the set of BIPA sounds:
     `{'dz', 'dzː', 'dz̪', 'dz̪ː', 'dzʰ', 'ˈʣʲ', 'ˈʣ', 'ⁿdz', 'ⁿdzʱ', 'dzʱ', 'dzʲ', 'dzˤ'}`

    Attributes:
        bipa (pyclts.TranscriptionSystem): CLTS's BIPA system
        change (dict): mapping of CLTS features to their coarse replacements.
            Replacements need not be CLTS features, but it makes things nicer if they are.
        remove (set): set of CLTS features to remove.
        labels (dict): mapping of coarse feature frozensets to the corresponding coarse sound string.
        cache (dict): direct mapping of BIPA sound strings to coarse sound string.
    """

    cache = {}

    def __init__(self, bipa, config):
        """
        Coarsen sounds from BIPA according to a configuation.

        At initialization, we compute the corresponding coarse feature sets for all listed
         BIPA sounds, assign a label to each, and cache this information.
         Other BIPA sounds may be encountered after initialization,
         for which the coarse sound will be computed on the fly.

        Example:
            >>> coarsen = Coarsen(clts.bipa, DEFAULT_CONFIG)
            >>> coarsen['dz̪']
            'dz'

        Args:
            bipa (pyclts.TranscriptionSystem): CLTS's BIPA system
            config (dict): a dictionnary of shape `{"change": dict, "remove": set}`

        """
        self.bipa = bipa
        self.change = config["change"]
        self.remove = config["remove"]
        self.labels = {} # coarse feature set -> coarse string
        self.cache = {} # bipa string -> coarse string

        # Construct a dict of coarse feature set -> all listed bipa sounds resulting in this set
        sounds = defaultdict(list)
        for sound in self.bipa.sounds:
            sound = self.bipa[sound]
            if isinstance(sound, COMPOSITE):
                coarse_f = self.coarsen_features(sound.from_sound.featureset)
                sounds[coarse_f].append(sound.from_sound)
                coarse_f = self.coarsen_features(sound.to_sound.featureset)
                sounds[coarse_f].append(sound.to_sound)
            else:
                coarse_f = self.coarsen_features(sound.featureset)
                sounds[coarse_f].append(sound)

        # Populate coarsened and cache
        for f in sounds:
            name = self._create_label(f,sounds[f])
            self.labels[f] = str(name)
            for s in sounds[f]:
                self.cache[str(s)] = str(name) # Instead, first try for the name

    def _create_label(self, features, bipa_sounds):
        """ Create a label for a coarse sound.

        A coarse sound is defined by a set of coarse features. Its label must be a
        BIPA sound which coarsens to this exact set of coarse features.
        In order to produce intuitive labels, we take the shortest string of:
            - The BIPA sound denoted by the coarse features,
                if it exists, and it that sound does coarsen to the sound into consideration.
            - The sound in `bipa_sounds` with the shortest BIPA string, or if identical,
            with the least features.

        For example:
            >>> self.create_label(frozenset({"approximant", "consonant",
            ...                 "anterior", "voiced"}), {"ɹ", "ɹ̩", "ð̞", "ɹː", "ɹʲ", "ɹʰ"})
            'ɹ'

        Args:
            features (frozenset): coarse feature set which defines a coarse sound.
            bipa_sounds (iterable of str): iterable of bipa strings which result in this coarse sound.

        Returns: a bipa sound which should serve as an alias for this feature set.
        """
        categories = {"consonant", "vowel", "cluster", "diphthong", "tone"}
        cat = categories & features
        other_features = features - cat
        feature_str = " ".join(other_features) + " " + " ".join(cat)
        candidates = []
        try:
            bipa_label = self.bipa[feature_str]
            f = bipa_label.featureset
            if bipa_label in bipa_sounds or self.coarsen_features(f) == features:
                candidates.append(bipa_label)
        except: pass
        short_label = min(bipa_sounds, key=lambda s:(len(str(s)), len(s.featureset)))
        candidates.append(short_label)
        return min(candidates, key=lambda s:(len(str(s)),len(s.featureset)))

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
                raise ValueError("Unknown sound "+item)
            if isinstance(sound, COMPOSITE):
                sa = self.coarsen_sound(sound.from_sound)
                sb = self.coarsen_sound(sound.to_sound)
                new_sound = self.bipa[sa+sb]
                if not isinstance(new_sound, COMPOSITE + (pyclts.models.UnknownSound,)):
                    # we reduced this into a simple sound, needs further coarsening
                    coarse_sound = self[str(new_sound)]
                else:
                    coarse_sound = sa+sb
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
        f = self.coarsen_features(simple_sound.featureset)
        try:
            return self.labels[f]
        except KeyError:
            self.labels[f] = self._create_label(f, {simple_sound})
            return str(simple_sound)

    def coarsen_features(self, featureset):
        """ Coarsen BIPA features.

        Args:
            featureset (frozenset): BIPA features defining a BIPA sound.

        Returns:
            coarse (frozenset): Coarse features defining a coarse sound.

        """
        features = set(featureset) - self.remove
        for f in list(features):
            if f in self.change:
                features.remove(f)
                features.add(self.change[f])
        return frozenset(features)

    def as_table(self):
        """ Describe all known sounds as a table, ready for csv export.

        Creates a table as a list of rows. The table includes a header.
        Each row is alist of strings and represents a coarse sound.
        The columns are:
            - "BIPA": the list of known BIPA sounds which result in this coarse sound.
            - "Coarse": the label of this coarse sound
            - "Coarse features": the features which define this coarse sound

        Returns:
            rows (list of list): list of known coarse sounds and their BIPA counterparts.
        """
        reversed_cache = defaultdict(list)
        for bipa in self.cache:
            coarse = self.cache[bipa]
            reversed_cache[coarse].append(bipa)
        rows = [["BIPA","Coarse","Coarse features"]]
        for fs in self.labels:
            coarse = self.labels[fs]
            all_bipa = reversed_cache[coarse]
            rows.append([" ".join(all_bipa),coarse, " ".join(fs)])
        return rows
