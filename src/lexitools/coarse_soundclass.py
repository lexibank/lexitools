#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coarsen sound classes"""
from pyclts import CLTS
from collections import defaultdict

class Coarsen(object):

    cache = {}

    def __init__(self, bipa, config):
        self.bipa = bipa
        self.change = config["change"]
        self.remove = config["remove"]
        self.coarsened = {}
        self.cache = {}

        sounds = defaultdict(list)
        for sound in self.bipa.sounds:
            print(sound)
            coarse_f = self.features(sound)
            sounds[coarse_f].append(sound)
        for f in sounds:
            name = min(sounds[f], key=lambda s:len(self.bipa[s].featureset))
            self.coarsened[f] = str(name)

    def __getitem__(self, item):
        try:
            return self.cache[item]
        except KeyError:
            features = self.features(item)
            coarse_sound = self.coarsened[features]
            self.cache[item] = coarse_sound
            return coarse_sound

    def features(self, item):
        sound = self.bipa[item]
        if sound.type == "unknownsound":
            return set()
        features = set(sound.featureset) - self.remove
        for f in list(features):
            if f in self.change:
                features.remove(f)
                features.add(self.change[f])
        return " ".join(sorted(features))


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
            'with_mid_tone', 'with_rising_tone', 'with_upstep'},
    change={'alveolar': 'anterior', 'alveolo-palatal': 'palatal', 'close-mid': 'mid',
          'dental': 'anterior', 'linguolabial': 'labial', 'nasal-click': 'click',
          'near-back': 'back', 'near-close': 'close', 'near-front': 'front',
          'near-open': 'open', 'open-mid': 'mid', 'palatal-velar': 'velar',
          'post-alveolar': 'palatal', 'tap': 'vibrant', 'trill': 'vibrant',
          'revoiced': 'voiced', 'devoiced': 'voiceless'})