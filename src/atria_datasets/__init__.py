from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_core.datasets import Dataset
    from atria_core.types import DataInstance, DatasetSplitType

    from .cr.nist_sd19 import nist_sd19 as nist_sd19
    from .htr.austrian_newspapers import austrian_newspapers as austrian_newspapers
    from .htr.bentham import bentham as bentham
    from .htr.bullinger import bullinger as bullinger
    from .htr.cvl import cvl as cvl
    from .htr.fhswf_german import fhswf_german_handwriting as fhswf_german_handwriting
    from .htr.german_kurrent_19c import german_kurrent_19c as german_kurrent_19c
    from .htr.gnhk import gnhk as gnhk
    from .htr.iam import iam as iam
    from .htr.iam_histdb import iam_histdb as iam_histdb
    from .htr.icdar2017_read_htr import (
        icdar2017_read_htr_a as icdar2017_read_htr_a,
        icdar2017_read_htr_b as icdar2017_read_htr_b,
    )
    from .htr.imgur5k import imgur5k as imgur5k
    from .htr.koenigsfelden import koenigsfelden as koenigsfelden
    from .htr.konzilsprotokolle import read_konzilsprotokolle as read_konzilsprotokolle
    from .htr.read_bozen import read_bozen as read_bozen
    from .htr.scads_german import (
        scadsai_german_handwriting as scadsai_german_handwriting,
    )
    from .htr.scads_german_fullpage import (
        scadsai_german_fullpage as scadsai_german_fullpage,
    )
    from .htr.stabs_ratsbuecher import stabs_ratsbuecher_o10 as stabs_ratsbuecher_o10
    from .qa.squad import squad as squad
    from .qa.textvqa import textvqa as textvqa


def load_dataset(
    name: str,
    data_dir: str | None = None,
    access_token: str | None = None,
    split: DatasetSplitType | None = None,
    **kwargs: Any,
) -> Dataset:
    """Build a named dataset with shared runtime and dataset-specific options."""
    if name not in __all__:
        available = ", ".join(sorted(__all__))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")
    factory = getattr(sys.modules[__name__], name)
    return factory(data_dir=data_dir, access_token=access_token, split=split, **kwargs)


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "cr.nist_sd19": ["nist_sd19"],
        "htr.austrian_newspapers": ["austrian_newspapers"],
        "htr.bentham": ["bentham"],
        "htr.bullinger": ["bullinger"],
        "htr.cvl": ["cvl"],
        "htr.fhswf_german": ["fhswf_german_handwriting"],
        "htr.german_kurrent_19c": ["german_kurrent_19c"],
        "htr.gnhk": ["gnhk"],
        "htr.iam": ["iam"],
        "htr.iam_histdb": ["iam_histdb"],
        "htr.icdar2017_read_htr": ["icdar2017_read_htr_a", "icdar2017_read_htr_b"],
        "htr.imgur5k": ["imgur5k"],
        "htr.koenigsfelden": ["koenigsfelden"],
        "htr.konzilsprotokolle": ["read_konzilsprotokolle"],
        "htr.read_bozen": ["read_bozen"],
        "htr.scads_german": ["scadsai_german_handwriting"],
        "htr.scads_german_fullpage": ["scadsai_german_fullpage"],
        "htr.stabs_ratsbuecher": ["stabs_ratsbuecher_o10"],
        "qa.squad": ["squad"],
        "qa.textvqa": ["textvqa"],
    },
)
