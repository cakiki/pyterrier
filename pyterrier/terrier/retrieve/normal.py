from ... import tqdm, check_version
from ...datasets import Dataset
from . base import BatchRetrieveBase, _parse_index_like, _from_dataset, _mergeDicts, _function2wmodel, _matchop
from typing import Union
from ...model import coerce_queries_dataframe, FIRST_RANK
import pandas as pd, numpy as np
from warnings import warn
import concurrent
from concurrent.futures import ThreadPoolExecutor

class BatchRetrieve(BatchRetrieveBase):
    """
    Use this class for retrieval by Terrier
    """

    @staticmethod
    def matchop(t, w=1):
        """
        Static method used for rewriting a query term to use a MatchOp operator if it contains
        anything except ASCII letters or digits.
        """
        import base64
        import string
        if not all(a in string.ascii_letters + string.digits for a in t):
            encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8") 
            t = f'#base64({encoded})'
        if w != 1:
            t = f'#combine:0={w}({t})'
        return t


    @staticmethod
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        """
        Instantiates a BatchRetrieve object from a pre-built index access via a dataset.
        Pre-built indices are ofen provided via the `Terrier Data Repository <http://data.terrier.org/>`_.

        Examples::

            dataset = pt.get_dataset("vaswani")
            bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
            #or
            bm25 = pt.BatchRetrieve.from_dataset("vaswani", "terrier_stemmed", wmodel="BM25")

        **Index Variants**:

        There are a number of standard index names.
         - `terrier_stemmed` - a classical index, removing Terrier's standard stopwords, and applying Porter's English stemmer
         - `terrier_stemmed_positions` - as per `terrier_stemmed`, but also containing position information
         - `terrier_unstemmed` - a classical index, without applying stopword removal or stemming
         - `terrier_stemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents
         - `terrier_unstemmed_text` - as per `terrier_stemmed`, but also containing the raw text of the documents

        """
        return _from_dataset(dataset, variant=variant, version=version, clz=BatchRetrieve, **kwargs)

    #: default_controls(dict): stores the default controls
    default_controls = {
        "terrierql": "on",
        "parsecontrols": "on",
        "parseql": "on",
        "applypipeline": "on",
        "localmatching": "on",
        "filters": "on",
        "decorate": "on",
        "wmodel": "DPH",
    }

    #: default_properties(dict): stores the default properties
    default_properties = {
        "querying.processes": "terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,context_wmodel:org.terrier.python.WmodelFromContextProcess,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on",
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope,applypipeline",
        "termpipelines": "Stopwords,PorterStemmer"
    }

    def __init__(self, index_location, controls=None, properties=None, metadata=["docno"],  num_results=None, wmodel=None, threads=1, **kwargs):
        """
            Init method

            Args:
                index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
                controls(dict): A dictionary with the control names and values
                properties(dict): A dictionary with the property keys and values
                verbose(bool): If True transform method will display progress
                num_results(int): Number of results to retrieve. 
                metadata(list): What metadata to retrieve
        """
        super().__init__(kwargs)
        from ... import autoclass
        self.indexref = _parse_index_like(index_location)
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')
        self.properties = _mergeDicts(BatchRetrieve.default_properties, properties)
        self.concurrentIL = autoclass("org.terrier.structures.ConcurrentIndexLoader")
        if check_version(5.5) and "SimpleDecorateProcess" not in self.properties["querying.processes"]:
            self.properties["querying.processes"] += ",decorate:SimpleDecorateProcess"
        self.metadata = metadata
        self.threads = threads
        self.RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")
        self.search_context = {}

        for key, value in self.properties.items():
            self.appSetup.setProperty(str(key), str(value))
        
        self.controls = _mergeDicts(BatchRetrieve.default_controls, controls)
        if wmodel is not None:
            from ...transformer import is_lambda, is_function
            if isinstance(wmodel, str):
                self.controls["wmodel"] = wmodel
            elif is_lambda(wmodel) or is_function(wmodel):
                callback, wmodelinstance = _function2wmodel(wmodel)
                #save the callback instance in this object to prevent being GCd by Python
                self._callback = callback
                self.search_context['context_wmodel'] = wmodelinstance
                self.controls['context_wmodel'] = 'on'
            elif isinstance(wmodel, autoclass("org.terrier.matching.models.WeightingModel")):
                self.search_context['context_wmodel'] = wmodel
                self.controls['context_wmodel'] = 'on'
            else:
                raise ValueError("Unknown parameter type passed for wmodel argument: %s" % str(wmodel))
                  
        if self.threads > 1:
            warn("Multi-threaded retrieval is experimental, YMMV.")
            assert check_version(5.5), "Terrier 5.5 is required for multi-threaded retrieval"

            # we need to see if our indexref is concurrent. if not, we upgrade it using ConcurrentIndexLoader
            # this will upgrade the underlying index too.
            if not self.concurrentIL.isConcurrent(self.indexref):
                warn("Upgrading indexref %s to be concurrent" % self.indexref.toString())
                self.indexref = self.concurrentIL.makeConcurrent(self.indexref)

        if num_results is not None:
            if num_results > 0:
                self.controls["end"] = str(num_results -1)
            elif num_results == 0:
                del self.controls["end"]
            else: 
                raise ValueError("num_results must be None, 0 or positive")


        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")
        self.manager = MF._from_(self.indexref)
    
    def get_parameter(self, name : str):
        if name in self.controls:
            return self.controls[name]
        elif name in self.properties:
            return self.properties[name]
        else:
            return super().get_parameter(name)

    def set_parameter(self, name : str, value):
        if name in self.controls:
            self.controls[name] = value
        elif name in self.properties:
            self.properties[name] = value
        else:
            super().set_parameter(name,value)

    def __reduce__(self):
        return (
            self.__class__,
            (self.indexref,),
            self.__getstate__()
        )

    def __getstate__(self): 
        return  {
                'context' : self.search_context,
                'controls' : self.controls, 
                'properties' : self.properties, 
                'metadata' : self.metadata,
                }

    def __setstate__(self, d): 
        self.controls = d["controls"]
        self.metadata = d["metadata"]
        self.search_context = d["context"]
        self.properties.update(d["properties"])
        for key,value in d["properties"].items():
            self.appSetup.setProperty(key, str(value))

    def _retrieve_one(self, row, input_results=None, docno_provided=False, docid_provided=False, scores_provided=False):
        rank = FIRST_RANK
        qid = str(row.qid)
        query = row.query
        if len(query) == 0:
            warn("Skipping empty query for qid %s" % qid)
            return []

        srq = self.manager.newSearchRequest(qid, query)
        
        for control, value in self.controls.items():
            srq.setControl(control, str(value))

        for key, value in self.search_context.items():
            srq.setContextObject(key, value)

        # this is needed until terrier-core issue #106 lands
        if "applypipeline:off" in query:
            srq.setControl("applypipeline", "off")
            srq.setOriginalQuery(query.replace("applypipeline:off", ""))

        # transparently detect matchop queries
        if _matchop(query):
            srq.setControl("terrierql", "off")
            srq.setControl("parsecontrols", "off")
            srq.setControl("parseql", "off")
            srq.setControl("matchopql", "on")

        #ask decorate only to grab what we need
        srq.setControl("decorate", ",".join(self.metadata))

        # this handles the case that a candidate set of documents has been set. 
        num_expected = None
        if docno_provided or docid_provided:
            # we use RequestContextMatching to make a ResultSet from the 
            # documents in the candidate set. 
            matching_config_factory = self.RequestContextMatching.of(srq)
            input_query_results = input_results[input_results["qid"] == qid]
            num_expected = len(input_query_results)
            if docid_provided:
                matching_config_factory.fromDocids(input_query_results["docid"].values.tolist())
            elif docno_provided:
                matching_config_factory.fromDocnos(input_query_results["docno"].values.tolist())
            # batch retrieve is a scoring process that always overwrites the score; no need to provide scores as input
            #if scores_provided:
            #    matching_config_factory.withScores(input_query_results["score"].values.tolist())
            matching_config_factory.build()
            srq.setControl("matching", "org.terrier.matching.ScoringMatching" + "," + srq.getControl("matching"))

        # now ask Terrier to run the request
        self.manager.runSearchRequest(srq)
        result = srq.getResults()

        # check we got all of the expected metadata (if the resultset has a size at all)
        if len(result) > 0 and len(set(self.metadata) & set(result.getMetaKeys())) != len(self.metadata):
            raise KeyError("Mismatch between requested and available metadata in %s. Requested metadata: %s, available metadata %s" % 
                (str(self.indexref), str(self.metadata), str(result.getMetaKeys()))) 

        if num_expected is not None:
            assert(num_expected == len(result))
        
        rtr_rows=[]
        # prepare the dataframe for the results of the query
        for item in result:
            metadata_list = []
            for meta_column in self.metadata:
                metadata_list.append(item.getMetadata(meta_column))
            res = [qid, item.getDocid()] + metadata_list + [rank, item.getScore()]
            rank += 1
            rtr_rows.append(res)

        return rtr_rows


    def transform(self, queries):
        """
        Performs the retrieval

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        Returns:
            pandas.Dataframe with columns=['qid', 'docno', 'rank', 'score']
        """
        results=[]
        if not isinstance(queries, pd.DataFrame):
            warn(".transform() should be passed a dataframe. Use .search() to execute a single query.", FutureWarning, 2)
            queries = coerce_queries_dataframe(queries)
        
        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        input_results = None
        if docno_provided or docid_provided:
            assert check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            
        # make sure queries are a String
        if queries["qid"].dtype == np.int64:
            queries['qid'] = queries['qid'].astype(str)

        if self.threads > 1:

            if not self.concurrentIL.isConcurrent(self.indexref):
                raise ValueError("Threads must be set >1 in constructor and/or concurrent indexref used")
            
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                
                # we must detatch jnius to prevent thread leaks through JNI
                from jnius import detach
                def _one_row(*args, **kwargs):
                    rtr = self._retrieve_one(*args, **kwargs)
                    detach()
                    return rtr
                
                # create a future for each query, and submit to Terrier
                future_results = {
                    executor.submit(_one_row, row, input_results, docno_provided=docno_provided, docid_provided=docid_provided, scores_provided=scores_provided) : row.qid 
                    for row in queries.itertuples()}                
                
                # as these futures complete, wait and add their results
                iter = concurrent.futures.as_completed(future_results)
                if self.verbose:
                    iter = tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
                
                for future in iter:
                    res = future.result()
                    results.extend(res)
        else:
            iter = queries.itertuples()
            if self.verbose:
                iter = tqdm(iter, desc=str(self), total=queries.shape[0], unit="q")
            for row in iter:
                res = self._retrieve_one(row, input_results, docno_provided=docno_provided, docid_provided=docid_provided, scores_provided=scores_provided)
                results.extend(res)

        res_dt = pd.DataFrame(results, columns=['qid', 'docid' ] + self.metadata + ['rank', 'score'])
        # ensure to return the query and any other input columns
        input_cols = queries.columns[ (queries.columns == "qid") | (~queries.columns.isin(res_dt.columns))]
        res_dt = res_dt.merge(queries[input_cols], on=["qid"])
        return res_dt
        
    def __repr__(self):
        return "BR(" + ",".join([
            self.indexref.toString(),
            str(self.controls),
            str(self.properties)
            ]) + ")"

    def __str__(self):
        return "BR(" + self.controls["wmodel"] + ")"

    def setControls(self, controls):
        for key, value in controls.items():
            self.controls[str(key)] = str(value)

    def setControl(self, control, value):
        self.controls[str(control)] = str(value)