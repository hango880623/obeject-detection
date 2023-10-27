from typing import Dict, Union, Tuple

class RequestParams:
    base_url = 'https://maps.googleapis.com'

    def __init__(self, api_key: str, 
                        zoom: Union[str, int] = 20, 
                        img_size: Tuple[int, int] = (400,400), 
                        maptype: str = 'satellite', **kwargs) -> None:
        self._center = None
        self._zoom = str(zoom)
        self._size = '{}x{}'.format(img_size[0], img_size[1])
        self._maptype = maptype
        self._api_key = api_key
        self._url_params = {k: str(v).upper() for k, v in kwargs.items()}
        self._base = RequestParams.base_url

    @property
    def center(self) -> Dict[str, str]:
        return self._center

    @center.setter
    def center(self, value):
        self._center = value

    @center.deleter
    def center(self):
        del self._center

    def url_encoder(self) -> str:
        """
        Construct a url encoded request params
        :return: str
        """
        query_params = {key: str(val).strip().split() for key, val in self._url_params.items()}
        return self._url_encoder(**query_params)

    def _url_encoder(self, **kwargs) -> str: 
        """ 
        Url encoder helper method
        :params kwargs: center='foo', api_key='bar', or just api_key='bar', 
                        or an unpacked dict using ** operator
        :return: str
        """
        url_params = {k: v for k, v in kwargs.items()}
        for key in url_params:
            if len(url_params[key]) > 1:
                url_params[key] = '+'.join(url_params[key])
            else:
                url_params[key] = url_params[key][0]
        return ','.join([*url_params.values()])

    @property
    def url(self) -> str:
        url = '{}/maps/api/staticmap?'.format(self._base)
        self.center = self.url_encoder()
        return '{}center={}&zoom={}&size={}&maptype={}&key={}'.format(
            url, 
            self.center, 
            self._zoom, 
            self._size, 
            self._maptype, 
            self._api_key
        )