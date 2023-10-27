import requests, os
from typing import Optional
from dotenv import load_dotenv
from src.preprocess import RequestParams

class Client:
    load_dotenv()
    api_key = os.getenv('MAPS_KEY')

    def __init__(self, api_key: Optional[str] = None, **request_kwargs) -> None:
        """ 
        :param api_key: Google Maps API key. Required if not using the default key.
        :params **request_kwargs: a dict object containing street, city, state, and optional zipcode
        :return: None
        """
        self._api_key = api_key if api_key else Client.api_key
        self._request_kwargs = {k: str(v).upper() for k, v in request_kwargs.items()}

    @property
    def _query_params(self):
        return RequestParams(self._api_key, **self._request_kwargs)

    def _request(self):
        return requests.get(self._query_params.url)

    def response(self):
        response = self._request()
        if response.ok:
            return response # return response.json()
        else:
            print('fail to request')