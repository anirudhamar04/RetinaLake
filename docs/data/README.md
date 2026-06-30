# Datasets — layout & how to obtain them

RetinaLake does **not** redistribute any dataset. It ships the schema and the ingestion
adapters; you download each dataset from its original source under that dataset's own license
and terms of use, place it on disk under `STORAGE_DATA_ROOT`, then run the ingestion.

## On-disk layout

1. Obtain the raw datasets you need from their official sources (many require registration or a
   data-use agreement). Per-dataset notes and expected layouts are in the files in this folder.
2. Place each dataset under your `STORAGE_DATA_ROOT` (see `.env`) using the **exact folder name**
   in the catalogue below — the ingest adapters resolve `STORAGE_DATA_ROOT/<folder>` by that name.
3. Run the matching `chaksudb/ingest/scripts/ingest_NN_<name>.py` adapter, or
   `scripts/setup_full_database.py` to run the full pipeline.

So the on-disk layout looks like:

```
$STORAGE_DATA_ROOT/
├── 01_EYEPACS/
├── 02_MESSIDOR/
├── 03_IDRID/
└── …            # one folder per dataset, named exactly as in the catalogue
```

**You are responsible for complying with each dataset's license and usage restrictions.**

## Dataset catalogue

| #  | Dataset          | Folder name (`STORAGE_DATA_ROOT/…`) | Source / download                                                                                                                                             | Layout & notes                     |
| -- | ---------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 01 | EYEPACS          | `01_EYEPACS`                         | [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)                                                                                                | [docs](01_EYEPACS.md)       |
| 02 | MESSIDOR         | `02_MESSIDOR`                        | [ADCIS](https://www.adcis.net/en/third-party/messidor/)                                                                                                          | [docs](02_MESSIDOR.md)      |
| 03 | IDRiD            | `03_IDRID`                           | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)                                                           | [docs](03_IDRID.md)         |
| 04 | RFMiD            | `04_RFMid`                           | [IEEE DataPort](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)                                                          | [docs](04_RFMid.md)         |
| 05 | 1000×39 (JSIEC) | `05_1000x39`                         | [Sci. Reports](https://doi.org/10.1038/s41598-019-47181-w)                                                                                                       | [docs](05_1000x39.md)       |
| 06 | DeepEyeNet (DEN) | `06_DEN`                             | [GitHub](https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation)                             | [docs](06_DeepEyeNet.md)    |
| 07 | LAG              | `07_LAG`                             | [GitHub (AG-CNN)](https://github.com/smilell/AG-CNN)                                                                                                             | [docs](07_LAG_database.md)  |
| 08 | ODIR-5K          | `08_ODIR-5K`                         | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)                                                                            | [docs](08_ODIR5K.md)        |
| 09 | PAPILA           | `09_PAPILA`                          | [figshare](https://figshare.com/articles/dataset/PAPILA/14798004)                                                                                                | [docs](09_PAPILA.md)        |
| 10 | Paraguay         | `10_PARAGUAY`                        | [Zenodo](https://zenodo.org/record/4647952)                                                                                                                      | [docs](10_Paraguay.md)      |
| 11 | STARE            | `11_STARE`                           | [Clemson](https://cecas.clemson.edu/~ahoover/stare/)                                                                                                             | [docs](11_STARE.md)         |
| 12 | ARIA             | `12_ARIA`                            | —                                                                                                                                                                | [docs](12_ARIA.md)          |
| 13 | FIVES            | `13_FIVES`                           | [figshare](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)                                          | [docs](13_FIVES.md)         |
| 14 | AGAR300          | `14_AGAR300`                         | [IEEE DataPort](https://ieee-dataport.org/open-access/diabetic-retinopathy-fundus-image-datasetagar300)                                                          | [docs](14_AGAR_300.md)      |
| 15 | APTOS 2019       | `15_APTOS`                           | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)                                                                                                 | [docs](15_APTOS.md)         |
| 16 | FUND-OCT         | `16_FUND-OCT`                        | —                                                                                                                                                                | [docs](16_FUND-OCT.md)      |
| 17 | DiaRetDB1        | `17_DiaRetDB1`                       | [Kaggle (DiaRetDB1 v2.1)](https://www.kaggle.com/datasets/nguyenhung1903/diaretdb1-v21)                                                                          | [docs](17_DiaRetDB1.md)     |
| 18 | DRIONS-DB        | `18_DRIONS-DB`                       | [UNED](https://www.ia.uned.es/~ejcarmona/DRIONS-DB.html)                                                                                                         | [docs](18_DRIONS-DB.md)     |
| 19 | Drishti-GS1      | `19_Drishti-GS1`                     | [IIIT-H](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php)                                                                                  | [docs](19_Drishti-GS1.md)   |
| 20 | e-ophtha         | `20_E-ophta`                         | [ADCIS](https://www.adcis.net/en/third-party/e-ophtha/)                                                                                                          | [docs](20_E-ophta.md)       |
| 21 | G1020            | `21_G1020`                           | [arXiv:2006.09158](https://arxiv.org/abs/2006.09158)                                                                                                             | [docs](21_G1020.md)         |
| 22 | HEI-MED          | `22_HEI-MED`                         | [GitHub](https://github.com/lgiancaUTH/HEI-MED)                                                                                                                  | [docs](22_HEI-MED.md)       |
| 23 | HRF              | `23_HRF`                             | [FAU](https://www5.cs.fau.de/research/data/fundus-images/)                                                                                                       | [docs](23_HRF.md)           |
| 24 | ORIGA            | `24_ORIGA`                           | [Kaggle](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets)                                                                                           | [docs](24_ORIGA_light.md)   |
| 25 | REFUGE           | `25_REFUGE`                          | [IEEE DataPort](https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge)                                                                    | [docs](25_REFUGE.md)        |
| 26 | ROC              | `26_ROC`                             | [Univ. of Iowa](http://webeye.ophth.uiowa.edu/ROC/)                                                                                                              | [docs](26_ROC.md)           |
| 27 | BRSET            | `27_BRSET`                           | [PhysioNet](https://physionet.org/content/brazilian-ophthalmological/1.0.1)                                                                                      | [docs](27_BRSET.md)         |
| 28 | OIA-DDR          | `28_OIA-DDR`                         | [GitHub (DDR-dataset)](https://github.com/nkicsl/DDR-dataset)                                                                                                    | [docs](28_OIA-DDR.md)       |
| 29 | AIROGS           | `29_AIROGS`                          | [Zenodo](https://zenodo.org/records/5793241)                                                                                                                     | [docs](29_AIROGS.md)        |
| 30 | SUSTech-SYSU     | `30_SUSTech-SYSU`                    | [figshare](https://doi.org/10.6084/m9.figshare.12570770.v1)                                                                                                      | [docs](30_SUSTech-SYSU.md)  |
| 31 | JICHI            | `31_JICHI`                           | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179790)                                                                            | [docs](31_JICHI.md)         |
| 32 | Chaksu           | `32_CHAKSU`                          | [figshare](https://doi.org/10.6084/m9.figshare.20123135)                                                                                                         | [docs](32_Chakshu_IMAGE.md) |
| 33 | DR1-2            | `33_DR1-2`                           | [figshare (Pires et al.)](https://doi.org/10.6084/m9.figshare.953671.v3)                                                                                         | [docs](33_DR1-2.md)         |
| 34 | Cataract         | `34_Cataract`                        | [GitHub (cvblab)](https://github.com/cvblab/retina_dataset)                                                                                                      | [docs](34_Cataract.md)      |
| 35 | ScarDat          | `35_ScarDat`                         | [GitHub (Fundus10K)](https://github.com/li-xirong/fundus10k)                                                                                                     | [docs](35_ScarDat.md)       |
| 36 | ACRIMA           | `36_ACRIMA`                          | [figshare (Diaz-Pinto et al.)](https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135) | [docs](36_ACRIMA.md)        |
| 37 | DeepDRiD         | `37_DeepDRiD`                        | [GitHub](https://github.com/deepdrdoc/DeepDRiD)                                                                                                                  | [docs](37_DeepDRiD.md)      |
| 38 | MMAC             | `38_MMAC`                            | [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/12441)                                                                                                   | [docs](38_MMAC.md)          |
| 39 | JustRAIGS        | `39_justRAIGS`                       | [Zenodo](https://zenodo.org/records/10035093)                                                                                                                    | [docs](39_justRAIGS.md)     |
| 40 | RFMiD 2.0        | `40_RFMID2`                          | [IEEE DataPort](https://ieee-dataport.org/documents/retinal-fundus-multi-disease-image-dataset-rfmid-20)                                                         | —                         |
| 41 | CHASE-DB1        | `41_CHASEDB1`                        | [Kaggle](https://www.kaggle.com/datasets/khoongweihao/chasedb1)                                                                                                  | —                         |
| 42 | DRIVE            | `42_DRIVE`                           | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction)                                                           | —                         |
| 43 | DDR              | `43_DDR-dataset`                     | [GitHub](https://github.com/nkicsl/DDR-dataset)                                                                                                                  | —                         |
| 44 | RIM-ONE DL       | `44_RIM-ONE`                         | [GitHub](https://github.com/miag-ull/rim-one-dl)                                                                                                                 | —                         |
| 45 | RITE             | `45_RITE`                            | [Univ. of Iowa](https://medicine.uiowa.edu/eye/rite-dataset)                                                                                                     | —                         |
| 46 | MuReD            | `46_MuReD`                           | [Mendeley Data](https://data.mendeley.com/datasets/pc4mb3h8hz/1)                                                                                                 | —                         |
| 47 | MESSIDOR-2       | `47_MESSIDOR2`                       | [ADCIS](https://www.adcis.net/en/third-party/messidor2/)                                                                                                         | —                         |
| 48 | mBRSET           | `48_mbrset`                          | [PhysioNet](https://physionet.org/content/mbrset/1.0/)                                                                                                           | —                         |
| 49 | AV-DRIVE         | `49_AV_DRIVE`                        | [Univ. of Iowa (RITE)](https://medicine.uiowa.edu/eye/rite-dataset)                                                                                              | —                         |
| 50 | Fundus-AVSeg     | `50_Fundus-AVSeg`                    | [figshare](https://figshare.com/projects/Fundus-AVSeg_A_Fundus_Image_Dataset_for_AI-based_Artery-Vein_Vessel_Segmentation/229986)                                | —                         |
| 51 | HRF-v1           | `51_HRF-v1`                          | [GitHub (av-segmentation)](https://github.com/rubenhx/av-segmentation)                                                                                           | —                         |
| 52 | HRF-v2           | `52_HRF-v2`                          | [GitHub (av-segmentation)](https://github.com/rubenhx/av-segmentation)                                                                                           | —                         |
| 53 | LES-AV           | `53_LES-AV`                          | [figshare](https://figshare.com/articles/dataset/LES-AV_dataset/11857698)                                                                                        | —                         |
| 54 | MAPLES-DR        | `54_MAPLES`                          | [figshare](https://doi.org/10.6084/m9.figshare.24328660)                                                                                                         | —                         |

> Folder names are taken from each adapter's `get_data_root() / "<folder>"` resolution and the
> source links from its `DATASET_URL`. MESSIDOR-2 (47) and MAPLES-DR (54) re-annotate images that
> overlap with MESSIDOR (02), so install MESSIDOR (02) as well if you ingest them.
