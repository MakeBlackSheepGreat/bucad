# LSENS Reference Verification

Date: 2026-08-20

The eleven `\\bibitem` keys in the LSENS manuscript were matched one-to-one with `.neural-paper/references.csv`.

| Keys | Verification source | Result |
| --- | --- | --- |
| bray2024, gomez2024, liu2022convnext, liu2021swin, shin2019, kim2021, wang2024, busi2020 | Crossref works API queried by DOI | Title, author family names, publication year, venue, volume/page or article number, and DOI checked. |
| busiwhu2023, tcia2024 | DataCite DOI API queried by DOI | Dataset title, creators, publisher, year, version where applicable, resource type, and DOI checked. |
| efron1994 | Crossref works API queried by DOI | Book title, authors, publisher, year, and DOI checked. |

Verification notes: BUS-BRA has a 2023 Crossref issued date and a 2024 Medical Physics print citation; the manuscript uses the journal volume, issue, pages, and 2024 print year. TCIA's current collection page identifies the BrEaST dataset as version 2 and provides the DataCite DOI `10.7937/9WKK-Q141`.

Final recheck: all DOI queries were repeated after the final content revision. BUS-BRA's Crossref abstract explicitly confirms 1,875 images, 1,064 female patients, four ultrasound scanners, biopsy-proven labels, and BI-RADS categories 2--5. Its citation is used for the dataset and protocol statement. The acquisition-variability statement in the introduction is general modality context and is not attributed to the dataset citation.
