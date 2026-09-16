from __future__ import annotations

from atria_core.datasets import datasets

_DATASET_IMPORT_PATHS: dict[str, str] = {
    "austrian_newspapers": "atria_datasets.htr.austrian_newspapers.AustrianNewspapers",
    "bentham": "atria_datasets.htr.bentham.Bentham",
    "bullinger": "atria_datasets.htr.bullinger.Bullinger",
    "cvl": "atria_datasets.htr.cvl.CVL",
    "fhswf_german_handwriting": "atria_datasets.htr.fhswf_german.FHSWFGermanHandwriting",
    "german_kurrent_19c": "atria_datasets.htr.german_kurrent_19c.GermanKurrent19C",
    "gnhk": "atria_datasets.htr.gnhk.GNHK",
    "iam": "atria_datasets.htr.iam.IAM",
    "iam_histdb": "atria_datasets.htr.iam_histdb.IAMHistDB",
    "icdar2017_read_htr_a": "atria_datasets.htr.icdar2017_read_htr.ICDAR2017ReadHTRA",
    "icdar2017_read_htr_b": "atria_datasets.htr.icdar2017_read_htr.ICDAR2017ReadHTRB",
    "imgur5k": "atria_datasets.htr.imgur5k.IMGUR5K",
    "koenigsfelden": "atria_datasets.htr.koenigsfelden.Koenigsfelden",
    # "mmlongbench_doc": "atria_datasets.qa.mmlongbench_doc.MMLongBenchDoc",
    # "mp_docvqa": "atria_datasets.qa.mp_docvqa.MPDocVQA",
    "nist_sd19": "atria_datasets.cr.nist_sd19.NISTSD19",
    "read_bozen": "atria_datasets.htr.read_bozen.ReadBozen",
    "read_konzilsprotokolle": "atria_datasets.htr.konzilsprotokolle.Konzilsprotokolle",
    "scadsai_german_fullpage": "atria_datasets.htr.scads_german_fullpage.ScaDSAIFullPage",
    "scadsai_german_handwriting": "atria_datasets.htr.scads_german.ScaDSAI",
    # "slidevqa": "atria_datasets.qa.slidevqa.SlideVQA",
    "squad": "atria_datasets.qa.squad.Squad",
    "stabs_ratsbuecher_o10": "atria_datasets.htr.stabs_ratsbuecher.StABSRatsbuecher",
    "textvqa": "atria_datasets.qa.textvqa.TextVqa",
}

datasets.from_dict(_DATASET_IMPORT_PATHS)


__all__ = ["datasets"]
