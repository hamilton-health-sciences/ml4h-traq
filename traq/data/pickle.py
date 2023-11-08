import os
import pickle
from typing import List, Sequence
from warnings import warn

import numpy as np
import pandas as pd

from .base import Plate, PlateType, Study, StudyID
from .exceptions import AmbiguousPlateException, SchemaException


class PickledPlate(Plate):
    """
    Represents a plate of a pickled study.
    """

    def __init__(
        self, plate_name: str, plate_data, var_ignore: Sequence[str] = tuple()
    ) -> None:
        """ """
        self.name = plate_name
        self._df, self._labels = plate_data

    def centres(self) -> List[StudyID]:
        raise NotImplementedError

    def participants(self) -> List[StudyID]:
        raise NotImplementedError

    def variables(self) -> List[StudyID]:
        raise NotImplementedError

    @property
    def type(self) -> PlateType:
        # TODO: this is too stringent
        if len(self._df) == len(np.unique(self._df.index)):
            return PlateType.single
        else:
            return PlateType.multi

    def to_frame(self) -> pd.DataFrame:
        return self._df

    @property
    def labels(self) -> np.array:
        return np.array(self._labels["any_change"].astype(int))


class PickledStudy(Study):
    """
    Represents a pickled study, which can optionally have corresponding anomaly
    labels per plate.
    """

    def __init__(self, pickle_path: str, var_ignore: Sequence[str] = tuple()) -> None:
        self.pickle_path = pickle_path

        self._data = pickle.load(open(pickle_path, "rb"))
        self._name = os.path.basename(pickle_path).split(".")[0]

        self._plates = []
        for plate_name in self._data:
            try:
                plate = PickledPlate(plate_name, self._data[plate_name], var_ignore)
                self._plates.append(plate)
            except (SchemaException, AmbiguousPlateException) as exc:
                warn(str(exc))

    def centres(self) -> List[StudyID]:
        raise NotImplementedError()

    def participants(self) -> List[StudyID]:
        raise NotImplementedError()

    def plates(self) -> List[PickledPlate]:
        return self._plates
