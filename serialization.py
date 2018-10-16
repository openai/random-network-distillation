import joblib

def get_ctor_kwargs(loc):
    """
    call this at the beginning of your constructor to get a dict,
    which you can use to reconstruct the object
    """
    del loc['self']
    return loc

class Serializable(object):
    def to_dict(self):
        """
        Return a dictionary
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """
        Create the instance using dictionary made by to_dict()
        """
        raise NotImplementedError

class SimpleSer(object):
    def to_dict(self):
        return {'obj' : self, 'ctor' : type(self)}

    @classmethod
    def from_dict(cls, d, new_scope=None): #pylint: disable=W0613
        return d['obj']

def from_dict(d, new_scope=None):
    cls = d['ctor']
    return cls.from_dict(d, new_scope=new_scope)


NPZ_NUM_FMT = '%.4i'

def to_files(obj, path):
    d = obj.to_dict()
    joblib.dump(d, path+'.jd', compress=1)

def from_files(path, new_scope=None):
    d = joblib.load(path)
    return from_dict(d, new_scope=new_scope)    

