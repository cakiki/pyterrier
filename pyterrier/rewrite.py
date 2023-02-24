import pyterrier as pt
import pandas as pd
from . import Transformer
from . import tqdm
from warnings import warn
from typing import List,Union
from types import FunctionType
from .index import TerrierTokeniser

from .terrier.rewrite import SDM, SequentialDependence, QueryExpansion, RM3, Bo1QueryExpansion, AxiomaticQE

def tokenise(tokeniser : Union[str,TerrierTokeniser,FunctionType] = 'english', matchop=False) -> Transformer:
    """

    Applies tokenisation to the query. By default, queries obtained from ``pt.get_dataset().get_topics()`` are
    normally tokenised.

    Args:
        tokeniser(Union[str,TerrierTokeniser,FunctionType]): Defines what tokeniser should be used - either a Java tokeniser name in Terrier, a TerrierTokeniser instance, or a function that takes a str as input and returns a list of str.
        matchop(bool): Whether query terms should be wrapped in matchops, to ensure they can be parsed by a Terrier BatchRetrieve transformer.
    
    Example - use default tokeniser::

        pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve()
        pipe.search("Question with 'capitals' and other stuff?")
    
    Example - roll your own tokeniser::

        poortokenisation = pt.rewrite.tokenise(lambda query: query.split(" ")) >> pt.BatchRetrieve()

    Example - for non-English languages, tokenise on standard UTF non-alphanumeric characters::

        utftokenised = pt.rewrite.tokenise(pt.TerrierTokeniser.utf)) >> pt.BatchRetrieve()
        utftokenised = pt.rewrite.tokenise("utf")) >> pt.BatchRetrieve()

    Example - tokenising queries using a `HuggingFace tokenizer <https://huggingface.co/docs/transformers/fast_tokenizers>`_ ::

        # this assumes the index was created in a pretokenised manner
        br = pt.BatchRetrieve(indexref)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        query_toks = pt.rewrite.tokenise(tok.tokenize, matchop=True)
        retr_pipe = query_toks >> br
    
    """
    _query_fn = None
    if isinstance(tokeniser, FunctionType):
        _query_fn = tokeniser
    else:
        tokeniser = TerrierTokeniser._to_obj(tokeniser)
        tokeniser = TerrierTokeniser._to_class(tokeniser)
        if "." not in tokeniser:
            tokeniser = 'org.terrier.indexing.tokenisation.' + tokeniser
        tokenobj = pt.autoclass(tokeniser)()
        _query_fn = tokenobj.getTokens

    def _join_str(input : Union[str,List[str]]):
        if isinstance(input, str):
            return input
        return ' '.join(input)

    def _join_str_matchop(input : List[str]):
        assert not isinstance(input, str), "Expected a list of strings"
        return ' '.join(map(pt.BatchRetrieve.matchop, input))
    
    if matchop:
        return pt.apply.query(lambda r: _join_str_matchop(_query_fn(r.query)))
    return pt.apply.query(lambda r: _join_str(_query_fn(r.query)))


def reset() -> Transformer:
    """
        Undoes a previous query rewriting operation. This results in the query formulation stored in the `"query_0"`
        attribute being moved to the `"query"` attribute, and, if present, the `"query_1"` being moved to
        `"query_0"` and so on. This transformation is useful if you have rewritten the query for the purposes
        of one retrieval stage, but wish a subquent transformer to be applies on the original formulation.

        Internally, this function applies `pt.model.pop_queries()`.

        Example::

            firststage = pt.rewrite.SDM() >> pt.BatchRetrieve(index, wmodel="DPH")
            secondstage = pyterrier_bert.cedr.CEDRPipeline()
            fullranker = firststage >> pt.rewrite.reset() >> secondstage

    """
    from .model import pop_queries
    return pt.apply.generic(lambda topics: pop_queries(topics))


def stash_results(clear=True) -> Transformer:
    """
    Stashes (saves) the current retrieved documents for each query into the column `"stashed_results_0"`.
    This means that they can be restored later by using `pt.rewrite.reset_results()`.
    thereby converting a retrieved documents dataframe into one of queries.

    Args: 
    clear(bool): whether to drop the document and retrieved document related columns. Defaults to True.

    """
    return _StashResults(clear)
    
def reset_results() -> Transformer:
    """
    Applies a transformer that undoes a `pt.rewrite.stash_results()` transformer, thereby restoring the
    ranked documents.
    """
    return _ResetResults()

class _StashResults(Transformer):

    def __init__(self, clear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear = clear

    def transform(self, topics_and_res: pd.DataFrame) -> pd.DataFrame:
        from .model import document_columns, query_columns
        if "stashed_results_0" in topics_and_res.columns:
            raise ValueError("Cannot apply pt.rewrite.stash_results() more than once")
        doc_cols = document_columns(topics_and_res)
        
        rtr =  []
        if self.clear:
            query_cols = query_columns(topics_and_res)            
            for qid, groupDf in topics_and_res.groupby("qid"):
                documentsDF = groupDf[doc_cols]
                queryDf = groupDf[query_cols].iloc[0]
                queryDict = queryDf.to_dict()
                queryDict["stashed_results_0"] = documentsDF.to_dict(orient='records')
                rtr.append(queryDict)
            return pd.DataFrame(rtr)
        else:
            for qid, groupDf in topics_and_res.groupby("qid"):
                groupDf = groupDf.reset_index().copy()
                documentsDF = groupDf[doc_cols]
                docsDict = documentsDF.to_dict(orient='records')
                groupDf["stashed_results_0"] = None
                for i in range(len(groupDf)):
                    groupDf.at[i, "stashed_results_0"] = docsDict
                rtr.append(groupDf)
            return pd.concat(rtr)   

    def __repr__(self):
        return "pt.rewrite.stash_results()"     

class _ResetResults(Transformer):

    def transform(self, topics_with_saved_docs : pd.DataFrame) -> pd.DataFrame:
        if "stashed_results_0" not in topics_with_saved_docs.columns:
            raise ValueError("Cannot apply pt.rewrite.reset_results() without pt.rewrite.stash_results() - column stashed_results_0 not found")
        from .model import query_columns
        query_cols = query_columns(topics_with_saved_docs)
        rtr = []
        for row in topics_with_saved_docs.itertuples():
            docsdf = pd.DataFrame.from_records(row.stashed_results_0)
            docsdf["qid"] = row.qid
            querydf = pd.DataFrame(data=[row])
            querydf.drop("stashed_results_0", axis=1, inplace=True)
            finaldf = querydf.merge(docsdf, on="qid")
            rtr.append(finaldf)
        return pd.concat(rtr)

    def __repr__(self):
        return "pt.rewrite.reset_results()"

def linear(weightCurrent : float, weightPrevious : float, format="terrierql", **kwargs) -> Transformer:
    """
    Applied to make a linear combination of the current and previous query formulation. The implementation
    is tied to the underlying query language used by the retrieval/re-ranker transformers. Two of Terrier's
    query language formats are supported by the `format` kwarg, namely `"terrierql"` and `"matchoptql"`. 
    Their exact respective formats are `detailed in the Terrier documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/querylanguage.md>`_.

    Args:
        weightCurrent(float): weight to apply to the current query formulation.
        weightPrevious(float): weight to apply to the previous query formulation.
        format(str): which query language to use to rewrite the queries, one of "terrierql" or "matchopql".

    Example::

        pipeTQL = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25, format="terrierql")
        pipeMQL = pt.apply.query(lambda row: "az") >> pt.rewrite.linear(0.75, 0.25, format="matchopql")
        pipeT.search("a")
        pipeM.search("a")

    Example outputs of `pipeTQL` and `pipeMQL` corresponding to the query "a" above:

    - Terrier QL output: `"(az)^0.750000 (a)^0.250000"`
    - MatchOp QL output: `"#combine:0:0.750000:1:0.250000(#combine(az) #combine(a))"`

    """
    return _LinearRewriteMix([weightCurrent, weightPrevious], format, **kwargs)

class _LinearRewriteMix(Transformer):

    def __init__(self, weights : List[float], format : str = 'terrierql', **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self.format = format
        if format not in ["terrierql", "matchopql"]:
            raise ValueError("Format must be one of 'terrierql', 'matchopql'")

    def _terrierql(self, row):
        return "(%s)^%f (%s)^%f" % (
            row["query_0"],
            self.weights[0],
            row["query_1"],
            self.weights[1])
    
    def _matchopql(self, row):
        return "#combine:0:%f:1:%f(#combine(%s) #combine(%s))" % (
            self.weights[0],
            self.weights[1],
            row["query_0"],
            row["query_1"])

    def transform(self, topics_and_res):
        from .model import push_queries
        
        fn = None
        if self.format == "terrierql":
            fn = self._terrierql
        elif self.format == "matchopql":
            fn = self._matchopql

        newDF = push_queries(topics_and_res)
        newDF["query"] = newDF.apply(fn, axis=1)
        return newDF

    def __repr__(self):
        return "pt.rewrite.linear()"