from collections import OrderedDict

from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

cash = OrderedDict(
    {
        ECOD: {},
        IForest: {},
        KNN: {},  # slow
        LOF: {},  # slow
        # OCSVM: {},  # very slow
        PCA: {},
        HBOS: {},
    }
)
