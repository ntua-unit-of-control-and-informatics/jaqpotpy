from sqlalchemy import create_engine
import pandas as pd
from typing import Union, List, Dict, Generator
from sqlalchemy import inspect


PATH = "jaqpotpy/helpers/periodic_table/elements.db"


class PeriodicTable(object):
    def __init__(self):
        self.__engine = create_engine("sqlite:///{path:s}".format(path=PATH), echo=False)

    @property
    def get_tables(self) -> List:
        inspector = inspect(self.__engine)
        return inspector.get_table_names()

    def _find_one(self, query) -> Dict:
        res = pd.read_sql_query(query, self.__engine)
        if len(res) > 1:
            raise Warning('Found more results for query. Returning the first one.')

        return res.loc[0].to_dict()

    def _find_many(self, query) -> Generator:
        res = pd.read_sql_query(query, self.__engine)

        for row in res.to_dict(orient='records'):
            yield row

    def _fetch_table(self, table, as_dataframe=False) -> Union[pd.DataFrame, Generator]:
        res = pd.read_sql_query('SELECT * FROM {}'.format(table), self.__engine)

        if as_dataframe:
            return res

        for row in res.to_dict(orient='records'):
            yield row