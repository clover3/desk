from typing import Any
import logging

from chair.table_lib import DictCache, TablePrintHelper
from taskman_client.named_number_proxy import NamedNumberProxy
from chair.tab_print import print_table
logger = logging.getLogger(__name__)


def print_scores_by_asking_server(
        condition_list, method_list,
        target_field, method_name_map, transpose=False):
    search = NamedNumberProxy()

    def load_method_relate_scores(method):
        ret = search.search_numbers(method)
        print(ret)
        d: dict[str, Any] = {}
        for e in ret:
            if e['field'] == target_field and e['name'] == method:
                key = e['condition']
                if key in d:
                    logger.warning("key=%s is duplicated. It was originally %s, replace with %s",
                                  str(key), str(d[key]), str(e['number']))
                d[e['condition']] = e['number']
        return d

    method_cache = DictCache(load_method_relate_scores)

    if not transpose:
        def get_score(row_key, col_key):
            per_method_d = method_cache.get_val(row_key)
            try:
                return per_method_d[col_key]
            except KeyError:
                return "-"

        row_head = "Model"
        printer = TablePrintHelper(
            condition_list,
            method_list,
            None,
            method_name_map,
            get_score,
            row_head,
        )
    else:
        def get_score(row_key, col_key):
            per_col_d = method_cache.get_val(col_key)
            try:
                return per_col_d[row_key]
            except KeyError:
                return "-"

        row_head = "Model"
        printer = TablePrintHelper(
            method_list,
            condition_list,
            method_name_map,
            None,
            get_score,
            row_head,
        )

    table = printer.get_table()
    print_table(table)
