# Maximum number of columns to be considered processable.
MIN_COLUMNS = 0
MAX_COLUMNS = 10_000

# A default combination of columns that uniquely identifies each row in a plate, along
# with some filters to identify additional possible indexers.
DEFAULT_ID_COLS = ["centre", "id"]
MAX_NUM_ID_COLS = 3
POTENTIAL_ID_COL_REGEXS = [
    r".+rn\d?$",  # report number in e.g. COMPASS, HIP-ATTACK - terminal # is optional
    r".+rptn\d?$",  # report number in e.g. MANAGE
    r"^dfseq$",  # iDataFax metadata column
    r"visit$",  # anything enumerating the visit is probably ok? (TODO check)
    r"report",  # `report_number` is used a handful of times in rely (TODO check)
]
FORCE_KEEP_COLS = ["dfseq"]  # always keep these metadata columns, when available

# Prefer typing for these columns out of the box.
TYPED_COLS = {
    "centre": "int",
    "id": "int",
    "dfseq": "int",
}

# Whether or not to include fields which indicate whether a record has been validated
# (such as HIPATTACK's `brevd`) in the input dataframe X. Technically, there is no
# leakage, but it could lead to over-optimistic estimates of performance if such fields
# are not available in future trials.
EXCLUDE_REVISION_INDICATOR_COLUMNS = True

# What fraction of fields in a record need to change for it to be considered a record-
# level change.
RECORD_CHANGE_FRACTION = 0.2

# If a column has a fraction of changes exceeding this, exclude that column from the
# changeset analysis.
EXCLUDE_COLUMN_CHANGE_FRAC = 0.25
