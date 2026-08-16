from __future__ import annotations

from atria_core.datasets._download._download_manager import UrlSpec
from atria_core.types import DatasetSplitType

from atria_datasets.htr._page_xml_dataset import PageXMLDataset


class ICDAR2017ReadHTRA(PageXMLDataset):
    """ICDAR2017 READ HTR traditional benchmark: the fully line-annotated
    Train-A and Test-A subsets."""

    __module_name__ = "icdar2017_read_htr_a"

    # Train-A and Test-A are the fully line-annotated HTR benchmark subsets.
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Train-A.tbz2?download=1",
            # Canonical spelling understood by Atria's archive extractor.
            url_ext=".tar.bz2",
            rel_output_file_path="Train-A",
        ),
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Test-A.tgz?download=1",
            url_ext=".tgz",
            rel_output_file_path="Test-A",
        ),
    ]
    description = (
        "Fully annotated Train-A/Test-A subsets of the ICDAR2017 READ HTR "
        "traditional benchmark."
    )
    homepage = "https://zenodo.org/records/835489"
    license_name = "CC BY 4.0"
    split_aliases = {
        DatasetSplitType.train: ("train-a", "train_a"),
        DatasetSplitType.test: ("test-a", "test_a"),
    }


class ICDAR2017ReadHTRB(PageXMLDataset):
    """ICDAR2017 READ HTR advanced benchmark: page-level Train-B and
    region-only Test-B1/Test-B2 subsets."""

    __module_name__ = "icdar2017_read_htr_b"

    # Train-B has page-level transcripts but no line geometry. Test-B1 and
    # Test-B2 expose regions only and are intended for end-to-end inference.
    urls = [
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Train-B_batch1.tbz2?download=1",
            url_ext=".tar.bz2",
            rel_output_file_path="Train-B_batch1",
        ),
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Train-B_batch2.tbz2?download=1",
            url_ext=".tar.bz2",
            rel_output_file_path="Train-B_batch2",
        ),
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Test-B1.tgz?download=1",
            url_ext=".tgz",
            rel_output_file_path="Test-B1",
        ),
        UrlSpec(
            url="https://zenodo.org/records/835489/files/Test-B2.tgz?download=1",
            url_ext=".tgz",
            rel_output_file_path="Test-B2",
        ),
    ]
    description = (
        "Page-level Train-B and region-only Test-B1/Test-B2 subsets of the "
        "ICDAR2017 READ HTR advanced benchmark."
    )
    homepage = "https://zenodo.org/records/835489"
    license_name = "CC BY 4.0"
    split_aliases = {
        DatasetSplitType.train: ("train-b", "train_b"),
        DatasetSplitType.validation: ("test-b1", "test_b1"),
        DatasetSplitType.test: ("test-b2", "test_b2"),
    }
