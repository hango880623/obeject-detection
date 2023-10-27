from typing import Dict, List, Union

GenericNumber = Union[int, float]
Key, Value = Union[str, int], Union[str, GenericNumber]
Prediction = Dict[Key, Value] # singular prediction
Detections = Dict[Key, List[Prediction]] # list of predicted objects detected per image
IndexMap = List[Union[str, int]]

class Postprocess:
    """
    Processes predictions resulting from POST request to flask api
    """
    def __init__(self) -> None:
        self._detection_map = None

    @property
    def detection_map(self):
        """ 
        setter method
        """
        return self._detection_map

    @detection_map.setter
    def detection_map(self, value) -> None: # value: Tuple[IndexMap, Detections]
        self._detection_map = value[0], list(value[1].values())

    @detection_map.deleter
    def detection_map(self) -> None:
        del self._detection_map

    def updated_detections(self): #-> Detections: Dict[str, List[Dict[str, Union[float, str, int]]]]:
        """ 
        returns predicted objects back in original nested dict form, filtering out
        observations which had no predictions
        """
        return {
            int(key): value for key, value in self._updated_detections().items() if len(value) > 0
            }

    def _updated_detections(self): #-> Detections:
        return dict(zip(self.detection_map[0], list(self.detection_map[-1])))

    def object_count(self):
        object_counts = {key: [] for key in self.updated_detections().keys()}
        for key, value in self.updated_detections().items():
            counts1, counts2 = 0, 0
            for prediction in value:
                if prediction['class'] == 3: # solar panel class id
                    counts1 += 1
                elif prediction['class'] == 2: # pool class id
                    counts2 += 1
            object_counts[key].append([counts1, counts2])
        return object_counts

    def prediction_summary(self, count: bool = False):
        """ 
        :params count: opts - [True, False], default: False 
                       True: returns dict with key: index of observation and value: dict(solar=0 or 1, pool = 0 or 1)
                       False: returns dict of keys: objects, values: class id (0,1,2,3), 
                       total count of each object detected, nested list of xmin_max, ymin_max
                       of each object, list of confidence of each prediction of that particular object, 

        """
        ## just simply returns yes or no/true or false for classes
        # doesn't return anything for indicies of observations that have no predictions

        pass