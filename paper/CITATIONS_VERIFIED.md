# CITATION VERIFICATION RECORD

Every reference in `adas_paper.tex` was checked against the publisher's own
proceedings listing (IEEE Xplore, CVF Open Access, AAAI OJS, ACM DL, or
NeurIPS Proceedings) — not against arXiv. Author lists, venue, year and page
ranges below are what those sources report.

| # | Key | Authors (as published) | Venue | Pages | Verified against |
|---|---|---|---|---|---|
| 1 | `geiger2012kitti` | A. Geiger, P. Lenz, R. Urtasun | IEEE CVPR 2012, Providence RI, Jun 16–21 | 3354–3361 | dblp `conf/cvpr/GeigerLU12` |
| 2 | `pan2018scnn` | X. Pan, J. Shi, P. Luo, X. Wang, X. Tang | AAAI 2018 | 7276–7283 | AAAI OJS (article 12301); dblp `conf/aaai/PanSLWT18` |
| 3 | `varma2019idd` | G. Varma, A. Subramanian, A. Namboodiri, M. Chandraker, C. V. Jawahar | IEEE WACV 2019 | 1743–1751 | IEEE DOI `10.1109/WACV.2019.00190` |
| 4 | `kumar2025driveindia` | R. Kumar, D. S. Reddy, P. Rajalakshmi | IEEE ITSC 2025 | — | Accepted at ITSC 2025; **see note A** |
| 5 | `iisc2025uvh26` | AIM, Indian Institute of Science | IISc Technical Report UVH-26-v1.0, Nov 2025 | — | IISc announcement; **see note B** |
| 6 | `redmon2016yolo` | J. Redmon, S. Divvala, R. Girshick, A. Farhadi | IEEE CVPR 2016, Las Vegas NV, Jun 27–30 | 779–788 | CVF Open Access, CVPR 2016 |
| 7 | `ultralytics2024yolo11` | G. Jocher, J. Qiu | Software release, Ultralytics 2024 | — | **see note C** |
| 8 | `zheng2022clrnet` | T. Zheng, Y. Huang, Y. Liu, W. Tang, Z. Yang, D. Cai, X. He | IEEE/CVF CVPR 2022 | 898–907 | CVF Open Access, CVPR 2022 |
| 9 | `qin2024ufldv2` | Z. Qin, P. Zhang, X. Li | IEEE TPAMI vol. 46, no. 5, May 2024 | 2555–2568 | IEEE DOI `10.1109/TPAMI.2022.3182097` |
| 10 | `zhang2022bytetrack` | Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang | ECCV 2022, LNCS vol. 13682, Springer | 1–21 | ACM DL `10.1007/978-3-031-20047-2_1` |
| 11 | `yang2024depthv2` | L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao | NeurIPS 2024, vol. 37 | — | `proceedings.neurips.cc` 2024 main track |
| 12 | `li2021zerodce` | C. Li, C. Guo, C. C. Loy | IEEE TPAMI vol. 44, no. 8, Aug 2022 | 4225–4238 | IEEE Xplore doc 9369102; DOI `10.1109/TPAMI.2021.3063604` |
| 13 | `lian2022geoconsistency` | Q. Lian, B. Ye, R. Xu, W. Yao, T. Zhang | IEEE/CVF CVPR 2022 | 1685–1694 | CVF Open Access, CVPR 2022 |
| 14 | `hartley2004multiple` | R. Hartley, A. Zisserman | *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge Univ. Press, 2004 | — | Standard monograph |

---

## Notes you must act on before submitting

**A — DriveIndia page numbers.** The paper is accepted at IEEE ITSC 2025 and the
proceedings are published, but the page range was not retrievable from open
sources at the time of writing. Look it up on IEEE Xplore and add
`pp.~XXX--YYY` to the bibliography entry. Everything else in that entry
(authors, venue, year) is confirmed.

**B — UVH-26 is a technical report, not a conference paper.** It has no peer-reviewed
venue as of writing; the only formal release is the IISc technical report. It is
cited as a technical report, which is its actual publication form — this is
accurate, not an arXiv substitution. If it is later accepted somewhere, update
the entry.

**C — YOLO11 has no peer-reviewed paper.** It is a software release, not a
publication, so it is cited as software with the repository URL, and the
*conceptual* contribution is credited to the original YOLO paper
(`redmon2016yolo`, CVPR 2016) which is peer-reviewed. Citing a non-existent
YOLOv11 paper would be a fabricated reference. Some reviewers prefer a version
number and access date on software citations — add `(accessed: DD Mon YYYY)`
if your target venue's style requires it.

---

## Author block — REQUIRES YOUR CORRECTION

The supervisor names were supplied as one unpunctuated string:
`"Pranshu Chnader Bhushan Singh Negi"`.

I have set this as **two** supervisors in the paper:
- Pranshu Chander
- Bhushan Singh Negi

`Chnader` is most likely a typo for `Chander` or `Chandra`. **Verify the correct
spelling and whether this is one person or two, and fix the `\author` block.**
Also fill the four `TODO` fields (institution, city, email) for each author.

---

## On originality

The paper text is written from scratch for this work. No passage is copied from
any source. The two quoted-in-substance findings (the domain-gap observation from
the IDD paper, and the reported adaptation gain from UVH-26) are paraphrased and
attributed inline with citations. All numerical results come from this project's
own measured runs — nothing is reproduced from another paper's results table.
