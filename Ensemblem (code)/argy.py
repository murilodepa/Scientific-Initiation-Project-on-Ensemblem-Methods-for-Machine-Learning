import sys

class Args:
    def __init__(self, sysargs=sys.argv):
        self.nested_params = {}
        self.args = {}
        if sysargs[0][-3:] == '.py':
            sysargs = sysargs[1:]
        for _p in sysargs:
            if _p[0] == '-':
                if _p in self.nested_params: raise KeyError("Duplicate argument found! ({})".format(_p))
                self.nested_params[_p] = []
                last_p = _p
            else:
                self.nested_params[last_p].append(_p)
        
    def add(self, param_alias, param_type=str, default_value=None):
        if param_alias in self.args:
            raise ValueError('Parameter {} already found in list'.format(param_alias))
        self.args[param_alias] = {'p_type': param_type, 'd_value': default_value}
        
    def get(self, param_alias, cut_one=True):
        if param_alias not in self.args:
            raise ValueError('Parameter {} not found in watchlist'.format(param_alias))
        if param_alias not in self.nested_params:
            return self.args[param_alias]['d_value']
        ret_list = []
        for _v in self.nested_params[param_alias]:
            ret_list.append(self.args[param_alias]['p_type'](_v))
        if cut_one and len(ret_list) == 1:
            return ret_list[0]
        return ret_list
        
    def getAllInputsAsList(self):
        return [k for k in self.nested_params]
