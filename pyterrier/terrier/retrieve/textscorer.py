from ... import Transformer
from .base import _parse_index_like
from .normal import BatchRetrieve

class TextIndexProcessor(Transformer):
    '''
        Creates a new MemoryIndex based on the contents of documents passed to it.
        It then creates a new instance of the innerclass and passes the topics to that.

        This class is the base class for TextScorer, but can be used in other settings as well, 
        for instance query expansion based on text.
    '''

    def __init__(self, innerclass, takes="queries", returns="docs", body_attr="body", background_index=None, verbose=False, **kwargs):
        #super().__init__(**kwargs)
        self.innerclass = innerclass
        self.takes = takes
        self.returns = returns
        self.body_attr = body_attr
        if background_index is not None:
            self.background_indexref = _parse_index_like(background_index)
        else:
            self.background_indexref = None
        self.kwargs = kwargs
        self.verbose = verbose

    def transform(self, topics_and_res):
        from ... import DFIndexer, autoclass, IndexFactory
        from ...index import IndexingType
        documents = topics_and_res[["docno", self.body_attr]].drop_duplicates(subset="docno")
        indexref = DFIndexer(None, type=IndexingType.MEMORY, verbose=self.verbose).index(documents[self.body_attr], documents["docno"])
        docno2docid = { docno:id for id, docno in enumerate(documents["docno"]) }
        index_docs = IndexFactory.of(indexref)
        docno2docid = {}
        for i in range(0, index_docs.getCollectionStatistics().getNumberOfDocuments()):
            docno2docid[index_docs.getMetaIndex().getItem("docno", i)] = i
        assert len(docno2docid) == index_docs.getCollectionStatistics().getNumberOfDocuments(), "docno2docid size (%d) doesnt match index (%d)" % (len(docno2docid), index_docs.getCollectionStatistics().getNumberOfDocuments())
        
        # if a background index is set, we create an "IndexWithBackground" using both that and our new index
        if self.background_indexref is None:
            index = index_docs
        else:
            index_background = IndexFactory.of(self.background_indexref)
            index = autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)          

        topics = topics_and_res[["qid", "query"]].dropna(axis=0, subset=["query"]).drop_duplicates()
        
        if self.takes == "queries":
            # we have provided the documents, so we dont need a docno or docid column that will confuse 
            # BR and think it is re-ranking. In fact, we only need qid and query
            input = topics
        elif self.takes == "docs":
            # we have to pass the documents, but its desirable to have the docids mapped to the new index already
            # build a mapping, as the metaindex may not have reverse lookups enabled
            input = topics_and_res.copy()
            # add the docid to the dataframe
            input["docid"] = input.apply(lambda row: docno2docid[row["docno"]], axis=1, result_type='reduce')


        # and then just instantiate BR using the our new index 
        # we take all other arguments as arguments for BR
        inner = self.innerclass(index, **(self.kwargs))
        inner.verbose = self.verbose
        inner_res = inner.transform(input)

        if self.returns == "docs":
            # as this is a new index, docids are not meaningful externally, so lets drop them
            inner_res.drop(columns=['docid'], inplace=True)

            topics_columns = topics_and_res.columns[(topics_and_res.columns.isin(["qid", "docno"])) | (~topics_and_res.columns.isin(inner_res.columns))]
            if len(inner_res) < len(topics_and_res):
                inner_res = topics_and_res[topics_columns].merge(inner_res, on=["qid", "docno"], how="left")
                inner_res["score"] = inner_res["score"].fillna(value=0)
            else:
                inner_res = topics_and_res[ topics_columns ].merge(inner_res, on=["qid", "docno"])
        elif self.returns == "queries":
            if len(inner_res) < len(topics):
                inner_res = topics.merge(on=["qid"], how="left")
        else:
            raise ValueError("returns attribute should be docs of queries")
        return inner_res

class TextScorer(TextIndexProcessor):
    """
        A re-ranker class, which takes the queries and the contents of documents, indexes the contents of the documents using a MemoryIndex, and performs ranking of those documents with respect to the queries.
        Unknown kwargs are passed to BatchRetrieve.

        Arguments:
            takes(str): configuration - what is needed as input: `"queries"`, or `"docs"`. Default is `"docs"` since v0.8.
            returns(str): configuration - what is needed as output: `"queries"`, or `"docs"`. Default is `"docs"`.
            body_attr(str): what dataframe input column contains the text of the document. Default is `"body"`.
            wmodel(str): example of configuration passed to BatchRetrieve.

        Example::

            df = pd.DataFrame(
                [
                    ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                    ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
                ], columns=["qid", "query", "text"])
            textscorer = pt.TextScorer(takes="docs", body_attr="text", wmodel="TF_IDF")
            rtr = textscorer.transform(df)
            #rtr will score each document for the query "chemical reactions" based on the provided document contents
    """

    def __init__(self, takes="docs", **kwargs):
        super().__init__(BatchRetrieve, takes=takes, **kwargs)
