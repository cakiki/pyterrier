from .normal import BatchRetrieve
from .base import _mergeDicts, _from_dataset, _matchop
from typing import Union
from warnings import warn
import pandas as pd, numpy as np
from ...datasets import Dataset
from ... import cast, tqdm, autoclass
from ...model import FIRST_RANK, coerce_queries_dataframe

class FeaturesBatchRetrieve(BatchRetrieve):
    """
    Use this class for retrieval with multiple features
    """

    #: FBR_default_controls(dict): stores the default properties for a FBR
    FBR_default_controls = BatchRetrieve.default_controls.copy()
    FBR_default_controls["matching"] = "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull"
    del FBR_default_controls["wmodel"]
    #: FBR_default_properties(dict): stores the default properties
    FBR_default_properties = BatchRetrieve.default_properties.copy()

    def __init__(self, index_location, features, controls=None, properties=None, threads=1, **kwargs):
        """
            Init method

            Args:
                index_location: An index-like object - An Index, an IndexRef, or a String that can be resolved to an IndexRef
                features(list): List of features to use
                controls(dict): A dictionary with the control names and values
                properties(dict): A dictionary with the property keys and values
                verbose(bool): If True transform method will display progress
                num_results(int): Number of results to retrieve. 
        """
        controls = _mergeDicts(FeaturesBatchRetrieve.FBR_default_controls, controls)
        properties = _mergeDicts(FeaturesBatchRetrieve.FBR_default_properties, properties)
        self.features = features
        properties["fat.featured.scoring.matching.features"] = ";".join(features)

        # record the weighting model
        self.wmodel = None
        if "wmodel" in kwargs:
            assert isinstance(kwargs["wmodel"], str), "Non-string weighting models not yet supported by FBR"
            self.wmodel = kwargs["wmodel"]
        if "wmodel" in controls:
            self.wmodel = controls["wmodel"]
        if threads > 1:
            raise ValueError("Multi-threaded retrieval not yet supported by FeaturesBatchRetrieve")
        
        super().__init__(index_location, controls, properties, **kwargs)

    def __reduce__(self):
        return (
            self.__class__,
            (self.indexref, self.features),
            self.__getstate__()
        )

    def __getstate__(self): 
        return  {
                'controls' : self.controls, 
                'properties' : self.properties, 
                'metadata' : self.metadata,
                'features' : self.features,
                'wmodel' : self.wmodel
                #TODO consider the context state?
                }

    def __setstate__(self, d): 
        self.controls = d["controls"]
        self.metadata = d["metadata"]
        self.features = d["features"]
        self.wmodel = d["wmodel"]
        self.properties.update(d["properties"])
        for key,value in d["properties"].items():
            self.appSetup.setProperty(key, str(value))
        #TODO consider the context state?

    @staticmethod 
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        return _from_dataset(dataset, variant=variant, version=version, clz=FeaturesBatchRetrieve, **kwargs)

    @staticmethod 
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):
        return _from_dataset(dataset, variant=variant, version=version, clz=FeaturesBatchRetrieve, **kwargs)

    def transform(self, queries):
        """
        Performs the retrieval with multiple features

        Args:
            queries: String for a single query, list of queries, or a pandas.Dataframe with columns=['qid', 'query']. For re-ranking,
                the DataFrame may also have a 'docid' and or 'docno' column.

        Returns:
            pandas.DataFrame with columns=['qid', 'docno', 'score', 'features']
        """
        results = []
        if not isinstance(queries, pd.DataFrame):
            warn(".transform() should be passed a dataframe. Use .search() to execute a single query.", FutureWarning, 2)
            queries = coerce_queries_dataframe(queries)

        docno_provided = "docno" in queries.columns
        docid_provided = "docid" in queries.columns
        scores_provided = "score" in queries.columns
        if docno_provided or docid_provided:
            #re-ranking mode
            from ... import check_version
            assert check_version(5.3)
            input_results = queries

            # query is optional, and functionally dependent on qid.
            # Hence as long as one row has the query for each qid, 
            # the rest can be None
            queries = input_results[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
            RequestContextMatching = autoclass("org.terrier.python.RequestContextMatching")

            if not scores_provided and self.wmodel is None:
                raise ValueError("We're in re-ranking mode, but input does not have scores, and wmodel is None")
        else:
            assert not scores_provided

            if self.wmodel is None:
                raise ValueError("We're in retrieval mode (input columns were "+str(queries.columns)+"), but wmodel is None. FeaturesBatchRetrieve requires a wmodel be set for identifying the candidate set. "
                    +" Hint: wmodel argument for FeaturesBatchRetrieve, e.g. FeaturesBatchRetrieve(index, features, wmodel=\"DPH\")")

        if queries["qid"].dtype == np.int64:
            queries['qid'] = queries['qid'].astype(str)

        newscores=[]
        for row in tqdm(queries.itertuples(), desc=str(self), total=queries.shape[0], unit="q") if self.verbose else queries.itertuples():
            qid = str(row.qid)
            query = row.query
            if len(query) == 0:
                warn("Skipping empty query for qid %s" % qid)
                continue

            srq = self.manager.newSearchRequest(qid, query)

            for control, value in self.controls.items():
                srq.setControl(control, str(value))

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

            # this handles the case that a candidate set of documents has been set. 
            if docno_provided or docid_provided:
                # we use RequestContextMatching to make a ResultSet from the 
                # documents in the candidate set. 
                matching_config_factory = RequestContextMatching.of(srq)
                input_query_results = input_results[input_results["qid"] == qid]
                if docid_provided:
                    matching_config_factory.fromDocids(input_query_results["docid"].values.tolist())
                elif docno_provided:
                    matching_config_factory.fromDocnos(input_query_results["docno"].values.tolist())
                if scores_provided:
                    if self.wmodel is None:
                        # we provide the scores, so dont use a weighting model, and pass the scores through Terrier
                        matching_config_factory.withScores(input_query_results["score"].values.tolist())
                        srq.setControl("wmodel", "Null")
                    else:
                        srq.setControl("wmodel", self.wmodel)
                matching_config_factory.build()
                srq.setControl("matching", ",".join(["FatFeaturedScoringMatching","ScoringMatchingWithFat", srq.getControl("matching")]))
            
            self.manager.runSearchRequest(srq)
            srq = cast('org.terrier.querying.Request', srq)
            fres = cast('org.terrier.learning.FeaturedResultSet', srq.getResultSet())
            feat_names = fres.getFeatureNames()

            docids=fres.getDocids()
            scores= fres.getScores()
            metadata_list = [fres.getMetaItems(meta_column) for meta_column in self.metadata]
            feats_values = [fres.getFeatureScores(feat) for feat in feat_names]
            rank = FIRST_RANK
            for i in range(fres.getResultSize()):
                doc_features = np.array([ feature[i] for feature in feats_values])
                meta=[ metadata_col[i] for metadata_col in metadata_list]
                results.append( [qid, query, docids[i], rank, doc_features ] + meta )
                newscores.append(scores[i])
                rank += 1

        res_dt = pd.DataFrame(results, columns=["qid", "query", "docid", "rank", "features"] + self.metadata)
        res_dt["score"] = newscores
        # ensure to return the query and any other input columns
        input_cols = queries.columns[ (queries.columns == "qid") | (~queries.columns.isin(res_dt.columns))]
        res_dt = res_dt.merge(queries[input_cols], on=["qid"])
        return res_dt

    def __repr__(self):
        return "FBR(" + ",".join([
            self.indexref.toString(),
            str(self.features),
            str(self.controls),
            str(self.properties)
        ]) + ")"

    def __str__(self):
        if self.wmodel is None:
            return "FBR(" + str(len(self.features)) + " features)"
        return "FBR(" + self.controls["wmodel"] + " and " + str(len(self.features)) + " features)"