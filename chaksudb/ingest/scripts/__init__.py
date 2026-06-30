"""
Ingest scripts module.

Exports all dataset ingestion functions for convenient importing.
"""

# Import all ingest functions
from chaksudb.ingest.scripts.ingest_01_eyepacs import ingest_eyepacs
from chaksudb.ingest.scripts.ingest_02_messidor import ingest_messidor
from chaksudb.ingest.scripts.ingest_03_idrid import ingest_idrid
from chaksudb.ingest.scripts.ingest_04_rfmid import ingest_rfmid
from chaksudb.ingest.scripts.ingest_05_1000x39 import ingest_1000x39
from chaksudb.ingest.scripts.ingest_06_den import ingest_deepeyenet
from chaksudb.ingest.scripts.ingest_07_lag import ingest_lag
from chaksudb.ingest.scripts.ingest_08_odir5k import ingest_odir5k
from chaksudb.ingest.scripts.ingest_09_papila import ingest_papila
from chaksudb.ingest.scripts.ingest_10_paraguay import ingest_paraguay
from chaksudb.ingest.scripts.ingest_11_stare import ingest_stare
from chaksudb.ingest.scripts.ingest_12_aria import ingest_aria
from chaksudb.ingest.scripts.ingest_13_fives import ingest_fives
from chaksudb.ingest.scripts.ingest_14_agar300 import ingest_agar300
from chaksudb.ingest.scripts.ingest_15_aptos import ingest_aptos
from chaksudb.ingest.scripts.ingest_16_fund_oct import ingest_fund_oct
from chaksudb.ingest.scripts.ingest_17_diaretdb1 import ingest_diaretdb1
from chaksudb.ingest.scripts.ingest_18_drionsdb import ingest_drionsdb
from chaksudb.ingest.scripts.ingest_19_drishti_gs1 import ingest_drishti_gs1
from chaksudb.ingest.scripts.ingest_20_eophta import ingest_eophta
from chaksudb.ingest.scripts.ingest_21_g1020 import ingest_g1020
from chaksudb.ingest.scripts.ingest_23_hrf import ingest_hrf
from chaksudb.ingest.scripts.ingest_24_origa import ingest_origa
from chaksudb.ingest.scripts.ingest_25_refuge import ingest_refuge
from chaksudb.ingest.scripts.ingest_26_roc import ingest_roc
from chaksudb.ingest.scripts.ingest_27_brset import ingest_brset
from chaksudb.ingest.scripts.ingest_28_oia_ddr import ingest_oia_ddr
from chaksudb.ingest.scripts.ingest_29_airogs import ingest_airogs
from chaksudb.ingest.scripts.ingest_30_sustech_sysu import ingest_sustech_sysu
from chaksudb.ingest.scripts.ingest_31_jichi import ingest_jichi
from chaksudb.ingest.scripts.ingest_32_chaksu import ingest_chaksu
from chaksudb.ingest.scripts.ingest_33_dr1_2 import ingest_dr1_2
from chaksudb.ingest.scripts.ingest_34_cataract import ingest_cataract
from chaksudb.ingest.scripts.ingest_35_scardat import ingest_scardat
from chaksudb.ingest.scripts.ingest_36_acrima import ingest_acrima
from chaksudb.ingest.scripts.ingest_37_deepdrid import ingest_deepdrid
from chaksudb.ingest.scripts.ingest_38_mmac import ingest_mmac
from chaksudb.ingest.scripts.ingest_22_hei_med import ingest_hei_med
from chaksudb.ingest.scripts.ingest_39_justraigs import ingest_justraigs
from chaksudb.ingest.scripts.ingest_40_rfmid2 import ingest_rfmid2
from chaksudb.ingest.scripts.ingest_41_chasedb1 import ingest_chasedb1
from chaksudb.ingest.scripts.ingest_42_drive import ingest_drive
from chaksudb.ingest.scripts.ingest_43_ddr import ingest_ddr
from chaksudb.ingest.scripts.ingest_44_rim_one import ingest_rim_one
from chaksudb.ingest.scripts.ingest_45_rite import ingest_rite
from chaksudb.ingest.scripts.ingest_46_mured import ingest_mured
from chaksudb.ingest.scripts.ingest_47_messidor2 import ingest_messidor2
from chaksudb.ingest.scripts.ingest_48_mbrset import ingest_mbrset
from chaksudb.ingest.scripts.ingest_49_av_drive import ingest_av_drive
from chaksudb.ingest.scripts.ingest_50_fundus_avseg import ingest_fundus_avseg
from chaksudb.ingest.scripts.ingest_51_hrf_v1 import ingest_hrf_v1
from chaksudb.ingest.scripts.ingest_52_hrf_v2 import ingest_hrf_v2
from chaksudb.ingest.scripts.ingest_53_les_av import ingest_les_av
from chaksudb.ingest.scripts.ingest_54_maples_dr import ingest_maples_dr

__all__ = [
    "ingest_eyepacs",
    "ingest_messidor",
    "ingest_idrid",
    "ingest_rfmid",
    "ingest_1000x39",
    "ingest_deepeyenet",
    "ingest_lag",
    "ingest_odir5k",
    "ingest_papila",
    "ingest_paraguay",
    "ingest_stare",
    "ingest_aria",
    "ingest_fives",
    "ingest_agar300",
    "ingest_aptos",
    "ingest_fund_oct",
    "ingest_diaretdb1",
    "ingest_drionsdb",
    "ingest_drishti_gs1",
    "ingest_eophta",
    "ingest_g1020",
    "ingest_hrf",
    "ingest_origa",
    "ingest_refuge",
    "ingest_roc",
    "ingest_brset",
    "ingest_oia_ddr",
    "ingest_airogs",
    "ingest_sustech_sysu",
    "ingest_jichi",
    "ingest_chaksu",
    "ingest_dr1_2",
    "ingest_cataract",
    "ingest_scardat",
    "ingest_acrima",
    "ingest_deepdrid",
    "ingest_mmac",
    "ingest_hei_med",
    "ingest_justraigs",
    "ingest_rfmid2",
    "ingest_chasedb1",
    "ingest_drive",
    "ingest_ddr",
    "ingest_rim_one",
    "ingest_rite",
    "ingest_mured",
    "ingest_messidor2",
    "ingest_mbrset",
    "ingest_av_drive",
    "ingest_fundus_avseg",
    "ingest_hrf_v1",
    "ingest_hrf_v2",
    "ingest_les_av",
    "ingest_maples_dr",
]
