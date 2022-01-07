"""
Constants.

Code used for constants.
"""

import pandas as pd

PATH_IMAGES = "data/images/"
PATH_PLOTS = "data/plots/"
PATH_HIHA_CEST = "data/data_from_rosenfeld/"

CUTOFF = [0.2, 20]

PATH_C3D = "~/.app/Contents/bin/c3d"


COLOR_LEFT = "#5252d6"
COLOR_RIGHT = "#d69452"
COLOR_ALL = "#777777"
COLOR_SUBJCTS = "#9f2626"
COLOR_CONTROLS = "#63269f"
COLOR_SUBJCTS_DARK = "#601717"
COLOR_CONTROLS_DARK = "#3c1760"

# %%
# =============================================================================
# Organization of imaging files
# =============================================================================

FILES = pd.DataFrame(columns=["sub", "group", "T1", "T2", "hipp", "cest"])

# controls
C004_T1 = PATH_IMAGES + "7T_C004/" + "t1.nii"
C004_T2 = PATH_IMAGES + "7T_C004/" + "layer_000_4f638b4ef023ed600443e41d2c34431e.nii.gz"
C004_HIPP = PATH_IMAGES + "7T_C004/" + "lr_segmentation.nii.gz"
C004_CEST = PATH_IMAGES + "7T_C004/" + "b0b1correctedcestmap.nii"

C005_T1 = PATH_IMAGES + "7T_C005/" + "layer_001_403a5f432fa45a55f89e53a015068436.nii.gz"
C005_T2 = PATH_IMAGES + "7T_C005/" + "layer_000_16989cc179621c3d6024c967e1953d3b.nii.gz"
C005_HIPP = PATH_IMAGES + "7T_C005/" + "lr_segmentation.nii.gz"
C005_CEST = PATH_IMAGES + "7T_C005/" + "b0b1correctedcestmap.nii"

C006_T1 = PATH_IMAGES + "7T_C006/" + "layer_001_9699b39aec13d873e044f4fa70580012.nii.gz"
C006_T2 = PATH_IMAGES + "7T_C006/" + "layer_000_becabf111f3dec8385f80a50181d42fd.nii.gz"
C006_HIPP = PATH_IMAGES + "7T_C006/" + "lr_segmentation.nii.gz"
C006_CEST = PATH_IMAGES + "7T_C006/" + "b0b1correctedcestmap.nii"

C007_T1 = PATH_IMAGES + "7T_C007/" + "layer_001_3c1f004395d13740e78315945ec893fe.nii.gz"
C007_T2 = PATH_IMAGES + "7T_C007/" + "layer_000_b3339e4bdd5b764c02e9358a0481829d.nii.gz"
C007_HIPP = PATH_IMAGES + "7T_C007/" + "lr_segmentation.nii.gz"
C007_CEST = PATH_IMAGES + "7T_C007/" + "b0b1correctedcestmap.nii"

C008_T1 = PATH_IMAGES + "7T_C008/" + "layer_001_8b223f92dea643aa74cadf64638367a0.nii.gz"
C008_T2 = PATH_IMAGES + "7T_C008/" + "layer_000_b9fc867f6891b0f05dc0df10273c4d62.nii.gz"
C008_HIPP = PATH_IMAGES + "7T_C008/" + "lr_segmentation.nii.gz"
C008_CEST = PATH_IMAGES + "7T_C008/" + "b0b1correctedcestmap.nii"

C009_T1 = PATH_IMAGES + "7T_C009/" + "layer_001_d988ec58977fbb57a14b4151a384ce68.nii.gz"
C009_T2 = PATH_IMAGES + "7T_C009/" + "layer_000_bc7c53105debdb7708d01483c150b023.nii.gz"
C009_HIPP = PATH_IMAGES + "7T_C009/" + "lr_segmentation.nii.gz"
C009_CEST = PATH_IMAGES + "7T_C009/" + "b0b1correctedcestmap.nii"

C010_T1 = PATH_IMAGES + "7T_C010/" + "layer_001_a5e5a2ad0956ae741369e28eed564a87.nii.gz"
C010_T2 = PATH_IMAGES + "7T_C010/" + "layer_000_09fb574abc000009960e5ee01e27f34e.nii.gz"
C010_HIPP = PATH_IMAGES + "7T_C010/" + "lr_segmentation.nii.gz"
C010_CEST = PATH_IMAGES + "7T_C010/" + "b0b1correctedcestmap.nii"

C011_T1 = PATH_IMAGES + "7T_C011/" + "layer_001_8b34f57814a2e60efbf291985ada2885.nii.gz"
C011_T2 = PATH_IMAGES + "7T_C011/" + "layer_000_59ecaee491aa109791c0795f94b1c933.nii.gz"
C011_HIPP = PATH_IMAGES + "7T_C011/" + "lr_segmentation.nii.gz"
C011_CEST = PATH_IMAGES + "7T_C011/" + "b0b1correctedcestmap.nii"

C012_T1 = PATH_IMAGES + "7T_C012/" + "layer_001_93b3670228c45de4c770ed4538e2fe07.nii.gz"
C012_T2 = PATH_IMAGES + "7T_C012/" + "layer_000_dab962cf2b0d44c5e4f313ab4b328785.nii.gz"
C012_HIPP = PATH_IMAGES + "7T_C012/" + "none"
C012_CEST = PATH_IMAGES + "7T_C012/" + "none"

C013_T1 = PATH_IMAGES + "7T_C013/" + "layer_001_4cd99df618ad0330e3cbedc9717f3485.nii.gz"
C013_T2 = PATH_IMAGES + "7T_C013/" + "layer_000_97a7216b625d77401847e3e9580af38f.nii.gz"
C013_HIPP = PATH_IMAGES + "7T_C013/" + "lr_segmentation.nii.gz"
C013_CEST = PATH_IMAGES + "7T_C013/" + "b0b1correctedcestmap.nii"

C014_T1 = PATH_IMAGES + "7T_C014/" + "layer_001_068624599fea610c57456a1c00063cc9.nii.gz"
C014_T2 = PATH_IMAGES + "7T_C014/" + "layer_000_d48f7ec52eac4292601fa81ecabf29e3.nii.gz"
C014_HIPP = PATH_IMAGES + "7T_C014/" + "lr_segmentation.nii.gz"
C014_CEST = PATH_IMAGES + "7T_C014/" + "b0b1correctedcestmap.nii"

C015_T1 = PATH_IMAGES + "7T_C015/" + "layer_001_48d52bccb276df856e86a8b2ca295a31.nii.gz"
C015_T2 = PATH_IMAGES + "7T_C015/" + "layer_000_7c56404e1bec4ff928283437945c3d08.nii.gz"
C015_HIPP = PATH_IMAGES + "7T_C015/" + "lr_segmentation.nii.gz"
C015_CEST = PATH_IMAGES + "7T_C015/" + "b0b1correctedcestmap.nii"

# patients
CHOP_03_T1 = (PATH_IMAGES + "CHOP_03/" + "layer_001_c42f2924309ddfdb0c95f0bea290a0d4.nii.gz")
CHOP_03_T2 = (PATH_IMAGES + "CHOP_03/" + "layer_000_c575dcceab4dec9c3d0969a2e89686a5.nii.gz")
CHOP_03_HIPP = PATH_IMAGES + "CHOP_03/" + "lr_segmentation.nii.gz"
CHOP_03_CEST = PATH_IMAGES + "CHOP_03/" + "b0b1correctedcestmap.nii"

CHOP_04_T1 = (PATH_IMAGES + "CHOP_04/" + "layer_001_24b2240414c09d557bd487348dfb71a8.nii.gz")
CHOP_04_T2 = (PATH_IMAGES + "CHOP_04/" + "layer_000_d3500cbd0c57e0169493d1555f13379f.nii.gz")
CHOP_04_HIPP = PATH_IMAGES + "CHOP_04/" + "lr_segmentation.nii.gz"
CHOP_04_CEST = PATH_IMAGES + "CHOP_04/" + "b0b1correctedcestmap.nii"

CHOP_05_T1 = (PATH_IMAGES + "CHOP_04_05/" + "none")
CHOP_05_T2 = (PATH_IMAGES + "CHOP_04_05/" + "T2_to_lr_segmentation.nii")
CHOP_05_HIPP = PATH_IMAGES + "CHOP_04_05/" + "lr_segmentation.nii.gz"
CHOP_05_CEST = PATH_IMAGES + "CHOP_04_05/" + "b0b1correctedcestmap.nii"

CHOP_06_T1 = (PATH_IMAGES + "CHOP_06/" + "layer_001_4c63ba68fefc646db8731b346f72ec3d.nii.gz")
CHOP_06_T2 = (PATH_IMAGES + "CHOP_06/" + "layer_000_d3610b6b9a31aafc00662c194e227173.nii.gz")
CHOP_06_HIPP = PATH_IMAGES + "CHOP_06/" + "lr_segmentation.nii.gz"
CHOP_06_CEST = PATH_IMAGES + "CHOP_06/" + "b0b1correctedcestmap.nii"

CHOP_07_T1 = (PATH_IMAGES + "CHOP_07/" + "layer_001_d2bd6de0aec1d2afa225721df465cff5.nii.gz")
CHOP_07_T2 = (PATH_IMAGES + "CHOP_07/" + "layer_000_5fc7fcde728514eacd3c56092d13b065.nii.gz")
CHOP_07_HIPP = PATH_IMAGES + "CHOP_07/" + "lr_segmentation.nii.gz"
CHOP_07_CEST = PATH_IMAGES + "CHOP_07/" + "b0b1correctedcestmap.nii"

CHOP_08_T1 = (PATH_IMAGES + "CHOP_08/" + "layer_001_56b6e3495f06baa92d7cc8251bc153fb.nii.gz")
CHOP_08_T2 = (PATH_IMAGES + "CHOP_08/" + "layer_000_c0455d4af5cde1fbdb27e751f86092c3.nii.gz")
CHOP_08_HIPP = PATH_IMAGES + "CHOP_08/" + "lr_segmentation.nii.gz"
CHOP_08_CEST = PATH_IMAGES + "CHOP_08/" + "b0b1correctedcestmap.nii"

CHOP_10_T1 = PATH_IMAGES + "CHOP_10/" + "none"
CHOP_10_T2 = PATH_IMAGES + "CHOP_10/" + "T2.nii"
CHOP_10_HIPP = PATH_IMAGES + "CHOP_10/" + "manual_segmentation_3.nii.gz"
CHOP_10_CEST = PATH_IMAGES + "CHOP_10/" + "b0b1correctedcestmap_10162019_S10.nii"

CHOP_11_T1 = (PATH_IMAGES + "CHOP_11/" + "layer_001_cb1833f1541ad81bb87f3ac517b9c844.nii.gz")
CHOP_11_T2 = (PATH_IMAGES + "CHOP_11/" + "layer_000_878ff7059c6b6b612cb280515e2c58b1.nii.gz")
CHOP_11_HIPP = PATH_IMAGES + "CHOP_11/" + "lr_segmentation.nii.gz"
CHOP_11_CEST = PATH_IMAGES + "CHOP_11/" + "b0b1correctedcestmap.nii"

CHOP_12_T1 = (PATH_IMAGES + "CHOP_12/" + "none")
CHOP_12_T2 = (PATH_IMAGES + "CHOP_12/" + "T2.nii")
CHOP_12_HIPP = PATH_IMAGES + "CHOP_12/" + "lr_segmentation.nii.gz"
CHOP_12_CEST = PATH_IMAGES + "CHOP_12/" + "b0b1correctedcestmap.nii"

# %%
NAMES_CONTROL = ["C004", "C005", "C006", "C007", "C008", "C009", "C010", "C011", "C013",
                 "C015"]
NAMES_PATIENTS = ["CHOP_03", "CHOP_04", "CHOP_05",
                  "CHOP_06", "CHOP_07", "CHOP_08", "CHOP_10", "CHOP_11", "CHOP_12"]


FILES = FILES.append(
    dict(
        sub="C004",
        group="control",
        T1=C004_T1,
        T2=C004_T2,
        hipp=C004_HIPP,
        cest=C004_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C005",
        group="control",
        T1=C005_T1,
        T2=C005_T2,
        hipp=C005_HIPP,
        cest=C005_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C006",
        group="control",
        T1=C006_T1,
        T2=C006_T2,
        hipp=C006_HIPP,
        cest=C006_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C007",
        group="control",
        T1=C007_T1,
        T2=C007_T2,
        hipp=C007_HIPP,
        cest=C007_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C008",
        group="control",
        T1=C008_T1,
        T2=C008_T2,
        hipp=C008_HIPP,
        cest=C008_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C009",
        group="control",
        T1=C009_T1,
        T2=C009_T2,
        hipp=C009_HIPP,
        cest=C009_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C010",
        group="control",
        T1=C010_T1,
        T2=C010_T2,
        hipp=C010_HIPP,
        cest=C010_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C011",
        group="control",
        T1=C011_T1,
        T2=C011_T2,
        hipp=C011_HIPP,
        cest=C011_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C013",
        group="control",
        T1=C013_T1,
        T2=C013_T2,
        hipp=C013_HIPP,
        cest=C013_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="C015",
        group="control",
        T1=C015_T1,
        T2=C015_T2,
        hipp=C015_HIPP,
        cest=C015_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_03",
        group="patient",
        T1=CHOP_03_T1,
        T2=CHOP_03_T2,
        hipp=CHOP_03_HIPP,
        cest=CHOP_03_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_04",
        group="patient",
        T1=CHOP_04_T1,
        T2=CHOP_04_T2,
        hipp=CHOP_04_HIPP,
        cest=CHOP_04_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_05",
        group="patient",
        T1=CHOP_05_T1,
        T2=CHOP_05_T2,
        hipp=CHOP_05_HIPP,
        cest=CHOP_05_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_06",
        group="patient",
        T1=CHOP_06_T1,
        T2=CHOP_06_T2,
        hipp=CHOP_06_HIPP,
        cest=CHOP_06_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_07",
        group="patient",
        T1=CHOP_07_T1,
        T2=CHOP_07_T2,
        hipp=CHOP_07_HIPP,
        cest=CHOP_07_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_08",
        group="patient",
        T1=CHOP_08_T1,
        T2=CHOP_08_T2,
        hipp=CHOP_08_HIPP,
        cest=CHOP_08_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_10",
        group="patient",
        T1=CHOP_10_T1,
        T2=CHOP_10_T2,
        hipp=CHOP_10_HIPP,
        cest=CHOP_10_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_11",
        group="patient",
        T1=CHOP_11_T1,
        T2=CHOP_11_T2,
        hipp=CHOP_11_HIPP,
        cest=CHOP_11_CEST,
    ),
    ignore_index=True,
)
FILES = FILES.append(
    dict(
        sub="CHOP_12",
        group="patient",
        T1=CHOP_12_T1,
        T2=CHOP_12_T2,
        hipp=CHOP_12_HIPP,
        cest=CHOP_12_CEST,
    ),
    ignore_index=True,
)
