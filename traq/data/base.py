from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import pandas as pd
from pandas._typing import DtypeObj

StudyID = int


class PlateType(Enum):
    """
    Possible types of plates.
    """

    # A single plate is a plate which has rows that are guaranteed to be one-to-
    # one with participants, i.e. representing a CRF that is only collected once
    # per participant.
    single = "single"

    # A multi plate is a plate which can have more than one row per participant.
    multi = "multi"


class Variable(ABC):
    """
    Represents a variable collected in a plate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the variable.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> DtypeObj:
        """
        The datatype of the variable.
        """
        pass


class Plate(ABC):
    """
    Represents a plate at a particular point in time.
    """

    @abstractmethod
    def type(self) -> PlateType:
        """
        Return the type of the plate.
        """
        pass

    @abstractmethod
    def variables(self) -> List[Variable]:
        """
        Return the list of variables collected in the plate.
        """
        pass

    @abstractmethod
    def centres(self) -> List[StudyID]:
        """
        Return a list of the unique centre IDs in the plate.
        """
        pass

    @abstractmethod
    def participants(self) -> List[StudyID]:
        """
        Return a list of the unique participant IDs in the plate.
        """
        pass

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:
        """
        Return the plate as a data frame, indexed by centre and participant ID.
        """
        pass


class Study(ABC):
    """
    Represents a study at a particular point in time.
    """

    @abstractmethod
    def centres(self) -> List[StudyID]:
        """
        Return a list of the unique centre IDs for the study.
        """
        pass

    @abstractmethod
    def participants(self) -> List[StudyID]:
        """
        Return a list of the unique participant IDs for the study.
        """
        pass

    @abstractmethod
    def plates(self) -> List[Plate]:
        """
        Return a list of the plates that are part of the study.
        """
        pass
