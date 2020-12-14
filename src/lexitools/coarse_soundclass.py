#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coarsen sound classes"""
import pyclts
from collections import defaultdict

class Coarsen(object):

    cache = {}

    def __init__(self, bipa, config):
        self.bipa = bipa
        self.change = config["change"]
        self.remove = config["remove"]
        self.names = {} # coarse feature set -> coarse string
        self.cache = {} # bipa string -> coarse string

        # Construct a dict of coarse feature set -> all listed bipa sounds resulting in this set
        sounds = defaultdict(list)
        for sound in self.bipa.sounds:
            sound = self.bipa[sound]
            if isinstance(sound, (pyclts.models.Diphthong, pyclts.models.Cluster)):
                coarse_f = self.coarsen_features(sound.from_sound)
                sounds[coarse_f].append(sound.from_sound)
                coarse_f = self.coarsen_features(sound.to_sound)
                sounds[coarse_f].append(sound.to_sound)
            else:
                coarse_f = self.coarsen_features(sound)
                sounds[coarse_f].append(sound)

        # Populate coarsened and cache
        for f in sounds:
            name = min(sounds[f], key=lambda s:len(self.bipa[s].featureset))
            self.names[f] = str(name)
            for s in sounds[f]:
                self.cache[str(s)] = str(name)

    def __getitem__(self, item):
        try:
            return self.cache[item]
        except KeyError:
            sound = self.bipa[item]
            if isinstance(sound, pyclts.models.UnknownSound):
                raise ValueError("Unknown sound "+item)
            if isinstance(sound, (pyclts.models.Diphthong, pyclts.models.Cluster)):
                sa = self.coarsen_sound(sound.from_sound)
                sb = self.coarsen_sound(sound.to_sound)
                coarse_sound = sa + sb
            else:
                coarse_sound = self.coarsen_sound(sound)
            self.cache[item] = coarse_sound
            return coarse_sound

    def coarsen_sound(self, simple_sound):
        f = self.coarsen_features(simple_sound)  # simple or complex sound
        try:
            return self.names[f]
        except KeyError:
            self.names[f] = str(simple_sound)
            return str(simple_sound)

    def coarsen_features(self, sound):
        features = set(sound.featureset) - self.remove
        for f in list(features):
            if f in self.change:
                features.remove(f)
                features.add(self.change[f])
        return frozenset(features)

DEFAULT_CONFIG = dict(
    remove={'advanced', 'advanced-tongue-root', 'apical', 'aspirated', 'breathy',
            'centralized', 'creaky', 'ejective', 'glottalized', 'labialized',
            'labio-palatalized', 'laminal', 'less-rounded', 'long', 'lowered',
            'mid-centralized', 'mid-long', 'more-rounded', 'non-syllabic',
            # 'nasalized'
            'palatalized', 'pharyngealized', 'pre-aspirated', 'pre-glottalized',
            'pre-labialized', 'pre-nasalized', 'pre-palatalized', 'primary-stress',
            'raised', 'retracted', 'retracted-tongue-root', 'rhotacized',
            'secondary-stress', 'strong', 'syllabic', 'ultra-long', 'ultra-short',
            'unreleased', 'velarized', 'with-frication', 'with-lateral-release',
            'with-mid-central-vowel-release', 'with-nasal-release', 'with_downstep',
            'with_extra-high_tone', 'with_extra-low_tone', 'with_falling_tone',
            'with_global_fall', 'with_global_rise', 'with_high_tone', 'with_low_tone',
            'with_mid_tone', 'with_rising_tone', 'with_upstep',
            'revoiced', 'devoiced'},
    change={'alveolar': 'anterior', 'alveolo-palatal': 'palatal', 'close-mid': 'mid',
          'dental': 'anterior', 'linguolabial': 'labial', 'nasal-click': 'click',
          'near-back': 'back', 'near-close': 'close', 'near-front': 'front',
          'near-open': 'open', 'open-mid': 'mid', 'palatal-velar': 'velar',
          'post-alveolar': 'palatal', 'tap': 'vibrant', 'trill': 'vibrant'})