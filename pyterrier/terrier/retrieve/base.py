from ... import tqdm, check_version, Transformer
from ...datasets import Dataset

from typing import Union
from ...transformer import Symbol


_matchops = ["#combine", "#uw", "#1", "#tag", "#prefix", "#band", "#base64", "#syn"]
def _matchop(query):
    for m in _matchops:
        if m in query:
            return True
    return False



def _mergeDicts(defaults, settings):
    KV = defaults.copy()
    if settings is not None and len(settings) > 0:
        KV.update(settings)
    return KV

def _function2wmodel(function):
    from ... import autoclass
    from jnius import PythonJavaClass, java_method

    class PythonWmodelFunction(PythonJavaClass):
        __javainterfaces__ = ['org/terrier/python/CallableWeightingModel$Callback']

        def __init__(self, fn):
            super(PythonWmodelFunction, self).__init__()
            self.fn = fn
            
        @java_method('(DLorg/terrier/structures/postings/Posting;Lorg/terrier/structures/EntryStatistics;Lorg/terrier/structures/CollectionStatistics;)D', name='score')
        def score(self, keyFreq, posting, entryStats, collStats):
            return self.fn(keyFreq, posting, entryStats, collStats)

        @java_method('()Ljava/nio/ByteBuffer;')
        def serializeFn(self):
            import dill as pickle
            #see https://github.com/SeldonIO/alibi/issues/447#issuecomment-881552005
            from dill import extend
            extend(use_dill=False)
            # keep both python and java representations around to prevent them being GCd in respective VMs 
            self.pbyterep = pickle.dumps(self.fn)
            self.jbyterep = autoclass("java.nio.ByteBuffer").wrap(self.pbyterep)
            return self.jbyterep

    callback = PythonWmodelFunction(function)
    wmodel = autoclass("org.terrier.python.CallableWeightingModel")( callback )
    return callback, wmodel

def _parse_index_like(index_location):
    from ... import autoclass, cast
    JIR = autoclass('org.terrier.querying.IndexRef')
    JI = autoclass('org.terrier.structures.Index')
    from ...index import TerrierIndexer

    if isinstance(index_location, JIR):
        return index_location
    if isinstance(index_location, JI):
        return cast('org.terrier.structures.Index', index_location).getIndexRef()
    if isinstance(index_location, str) or issubclass(type(index_location), TerrierIndexer):
        if issubclass(type(index_location), TerrierIndexer):
            return JIR.of(index_location.path)
        return JIR.of(index_location)

    raise ValueError(
        f'''index_location is current a {type(index_location)},
        while it needs to be an Index, an IndexRef, a string that can be
        resolved to an index location (e.g. path/to/index/data.properties),
        or an pyterrier.index.TerrierIndexer object'''
    )

class BatchRetrieveBase(Transformer, Symbol):
    """
    A base class for retrieval

    Attributes:
        verbose(bool): If True transform method will display progress
    """
    def __init__(self, verbose=0, **kwargs):
        super().__init__(kwargs)
        self.verbose = verbose

def _from_dataset(dataset : Union[str,Dataset], 
            clz,
            variant : str = None, 
            version='latest',            
            **kwargs) -> Transformer:

    from ... import get_dataset
    from ...io import autoopen
    import os
    import json
    
    if isinstance(dataset, str):
        dataset = get_dataset(dataset)
    if version != "latest":
        raise ValueError("index versioning not yet supported")
    indexref = dataset.get_index(variant)

    classname = clz.__name__
    # now look for, e.g., BatchRetrieve.args.json file, which will define the args for BatchRetrieve, e.g. stemming
    indexdir = indexref #os.path.dirname(indexref.toString())
    argsfile = os.path.join(indexdir, classname + ".args.json")
    if os.path.exists(argsfile):
        with autoopen(argsfile, "rt") as f:
            args = json.load(f)
            # anything specified in kwargs of this methods overrides the .args.json file
            args.update(kwargs)
            kwargs = args
    return clz(indexref, **kwargs)  