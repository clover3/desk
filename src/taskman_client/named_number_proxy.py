from taskman_client.RESTProxy import RESTProxy
from taskman_client.host_defs import webtool_host, webtool_port


class NamedNumberProxy:
    def __init__(self):
        self.server_proxy = RESTProxy(webtool_host, webtool_port)

    def search_numbers(self, keyword):
        return self.server_proxy.get(f"/experiment/named_number/?search={keyword}")

    def get_number(self, method, target_field, condition=None):
        ret = self.search_numbers(method)
        matched_number = None
        n_match = 0
        all_matched = []
        for e in ret:
            if e['field'] == target_field and e['name'] == method:
                if condition is None or condition == e['condition']:
                    matched_number = e["number"]
                    n_match += 1
                    all_matched.append(e)
        if n_match > 1:
            print(f"Warning {n_match} matched. :{all_matched}")
        return matched_number




