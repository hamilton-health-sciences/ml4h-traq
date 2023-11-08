class InputFileNotFound:
    message = "Input file not found."


class TRAQPreprocessingException(Exception):
    message = "A generic exception type."


class ExcludedTable(TRAQPreprocessingException):
    message = "This table has been excluded in the configuration."


class DataFrameLoadingException(TRAQPreprocessingException):
    message = "A data frame could not be loaded, for example because it has no columns."


class DataFrameTooLarge(TRAQPreprocessingException):
    message = (
        "Spark (Java) raised a StackOverflowError, likely indicating that the "
        "underlying data frame was too large."
    )

    def __init__(self, *args, **kwargs) -> None:
        self.original_exception = kwargs.pop("original_exception")


class PreliminarySnapshotTooWide(TRAQPreprocessingException):
    message = (
        "The preliminary snapshot's number of columns exceeds the configurable " "max."
    )


class GoldStandardSnapshotTooWide(TRAQPreprocessingException):
    message = (
        "The gold standard snapshot's number of columns exceeds the "
        "configurable max."
    )


class PreliminarySnapshotTooNarrow(TRAQPreprocessingException):
    message = (
        "The preliminary snapshot's number of columns is below the configurable " "min."
    )


class GoldStandardSnapshotTooNarrow(TRAQPreprocessingException):
    message = (
        "The gold standard snapshot's number of columns is below the "
        "configurable min."
    )


class EmptyTableSnapshot(TRAQPreprocessingException):
    message = "The table snapshot is empty (no rows)."


class GoldStandardTableNotFound(TRAQPreprocessingException):
    message = (
        "A table existed in a historical snapshot, but is not found in the final"
        " locked database."
    )


class NoMeaningfulData(TRAQPreprocessingException):
    message = "There are no non-metadata fields in the table."


class ParticipantIdentifierNotAvailable(TRAQPreprocessingException):
    message = "Either `centre` or `id` is not available in the preliminary table."


class SufficientIdentifiersNotAvailable(TRAQPreprocessingException):
    message = (
        "Unable to identify a combination of columns that uniquely identifies a row."
    )


class DuplicatesUncountable(TRAQPreprocessingException):
    message = "Duplicates are uncountable in the groupby statement."


class SnapshotSchemaChange(TRAQPreprocessingException):
    message = (
        "A schema change has occurred (such as a column changing type) "
        "between the preliminary snapshot and the gold-standard snapshot."
    )


class DiffIdentifiersInsufficient(TRAQPreprocessingException):
    message = (
        "The current set of identifiers (which was thought to be sufficient) is "
        "insufficient after coming out of the diff pipeline."
    )


class MetadataUnavailable(TRAQPreprocessingException):
    message = (
        "Metadata files which index available plates are unavailable for this trial."
    )
