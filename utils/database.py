from collections import deque, OrderedDict
import types
# import pyodbc
import sqlalchemy
from typing import List, Literal, Optional, Union, Dict, Hashable, Any
import toml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy.engine import URL


LOGGER = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
LOGGER.setLevel(logging.WARNING)

class Secrets(types.SimpleNamespace):
    """Singleton class containing login details"""
    def __init__(self, login_path: str="./.secrets/secrets.toml"):
        self.__SECRETS = toml.load(login_path)
        self.__dict__.update({k: self.__elt(v) for k, v in self.__SECRETS["database"].items()})
    
    def __elt(self, elt):
        """Recurse into elt to create leaf namespace objects"""
        if type(elt) is dict:
            return type(self)(elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Secrets, cls).__new__(cls)
            return cls.instance

SECRETS = Secrets()


class SQLDatabase:
    def __init__(
            self, 
            password: str=SECRETS.password, 
            username: str=SECRETS.username, 
            driver: str=SECRETS.driver,
            endpoint: str=SECRETS.endpoint, 
            database_name: str=SECRETS.database, 
            port: int=SECRETS.port):
        __conn_str = 'DRIVER='+driver+';SERVER=tcp:'+endpoint+';PORT='+str(port)+';DATABASE='+database_name+';UID='+username+';PWD='+password
        __conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": __conn_str})
        self.engine = sqlalchemy.create_engine(__conn_url)

    def query_to_pandas(self, query: str):
        with self.engine.connect().execution_options(autocommit=True) as conn:
            df = pd.read_sql(sqlalchemy.text(query), con=conn)
        return df

    @classmethod
    def to_pandas(cls, query: str, **kwargs):
        return cls(**kwargs).query_to_pandas(query)