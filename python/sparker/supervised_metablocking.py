import pyspark.sql.functions as f
from pyspark.sql.types import BooleanType


class SupervisedMB(object):

    @staticmethod
    def wnp(edges):
        """
        Performs the pruning by using the Weight Node Pruning.
        :param edges: DataFrame of edges with the probability column (p_match)
        """
        over_t = edges.filter("p_match >= 0.5")
        sc = edges.rdd.context
        p1_thresholds = sc.broadcast(over_t.select(["p1", "p_match"]).groupby("p1").mean().rdd.collectAsMap())
        p2_thresholds = sc.broadcast(over_t.select(["p2", "p_match"]).groupby("p2").mean().rdd.collectAsMap())
        
        @f.udf(returnType=BooleanType())
        def filter_wnp(p1, p2, w):
            p1_t = p1_thresholds.value.get(p1)
            p2_t = p2_thresholds.value.get(p2)
            return (w >= p1_t) or (w >= p2_t)
        
        pruned_edges = over_t.filter(filter_wnp('p1', 'p2', 'p_match'))
        return pruned_edges
    
    @staticmethod
    def rwnp(edges):
        """
        Performs the pruning by using the reciprocal Weight Node Pruning.
        :param edges: DataFrame of edges with the probability column (p_match)
        """
        over_t = edges.filter("p_match >= 0.5")
        sc = edges.rdd.context
        p1_thresholds = sc.broadcast(over_t.select(["p1", "p_match"]).groupby("p1").mean().rdd.collectAsMap())
        p2_thresholds = sc.broadcast(over_t.select(["p2", "p_match"]).groupby("p2").mean().rdd.collectAsMap())
        
        @f.udf(returnType=BooleanType())
        def filter_wnp(p1, p2, w):
            p1_t = p1_thresholds.value.get(p1)
            p2_t = p2_thresholds.value.get(p2)
            return (w >= p1_t) and (w >= p2_t)
        
        pruned_edges = over_t.filter(filter_wnp('p1', 'p2', 'p_match'))
        return pruned_edges

    @staticmethod
    def bcl(edges):
        """
        Performs the pruning by using the binary classifier outcome.
        :param edges: DataFrame of edges with the out column (pred)
        """
        return edges.filter("pred == 1")

    @staticmethod
    def cep(edges, blocks=None, blocks_sum_sizes=-1):
        """
        Performs the pruning by using the cardinality edge pruning method.
        :param edges: DataFrame of edges with the probability column (p_match)
        :param blocks: original block collection
        :param blocks_sum_sizes: sum of the sizes of the blocks in the block collection
        """
        over_t = edges.filter("p_match >= 0.5")
        
        if blocks is None and blocks_sum_sizes < 0:
            raise Exception("Provide blocks or their sum of sizes")
        
        if blocks_sum_sizes >= 0:
            number_of_edges_to_keep = int(blocks_sum_sizes/2)
        else:
            number_of_edges_to_keep = int(blocks.map(lambda b: b.get_size()).sum()/2)

        pruned_edges = over_t.sort('p_match', ascending=False).limit(number_of_edges_to_keep)
        return pruned_edges

    @staticmethod
    def wep(edges):
        """
        Performs the pruning by using the Weight Edge Pruning method
        :param edges: DataFrame of edges with the probability column (p_match)
        :return: dataframe with pruned edges
        """
        over_t = edges.filter("p_match >= 0.5")
        threshold = over_t.rdd.map(lambda x: x.asDict()['p_match']).mean()
        return over_t.filter(over_t.p_match >= threshold)

    @staticmethod
    def blast(edges):
        """
        Performs the pruning by using the BLAST method
        :param edges: DataFrame of edges with the probability column (p_match)
        :return: dataframe with pruned edges
        """
        sc = edges.rdd.context

        over_t = edges.filter("p_match >= 0.5")
        over_t.cache()
        over_t.count()

        profiles1_max_proba = sc.broadcast(over_t.groupby('p1').max('p_match').rdd.collectAsMap())
        profiles2_max_proba = sc.broadcast(over_t.groupby('p2').max('p_match').rdd.collectAsMap())

        def do_pruning(p1, p2, p_match):
            threshold = 0.35 * (profiles1_max_proba.value[p1] + profiles2_max_proba.value[p2])
            return p_match >= threshold

        pruning_udf = f.udf(do_pruning, BooleanType())

        res = over_t \
            .select("p1", "p2", "p_match", "is_match", pruning_udf("p1", "p2", "p_match").alias("keep")) \
            .where("keep") \
            .select("p1", "p2", "p_match", "is_match")
        res.count()
        over_t.unpersist()
        profiles1_max_proba.unpersist()
        profiles2_max_proba.unpersist()
        return res

    @staticmethod
    def rcnp(edges, profiles=None, n_entities=-1, blocks=None, blocks_sum_sizes=-1):
        """
        Performs the pruning by using the Reciprocal Cardinality Node Pruning method
        :param edges: DataFrame of edges with the probability column (p_match)
        :param profiles: RDD of profiles
        :param n_entities: number of profiles
        :param blocks: RDD of blocks
        :param blocks_sum_sizes: sum of the sizes of the blocks in the block collection
        :return: dataframe with pruned edges
        """
        
        if blocks is None and blocks_sum_sizes < 0:
            raise Exception("Provide blocks or their sum of sizes")
        
        if profiles is None and n_entities < 0:
            raise Exception("Provide profiles or their number")
        
        sc = edges.rdd.context

        over_t = edges.filter("p_match >= 0.5")
        over_t.cache()
        over_t.count()
        
        if n_entities < 0:
            n_entities = profiles.count()
        
        if blocks_sum_sizes >= 0:
            block_size = blocks_sum_sizes
        else:
            block_size = blocks.map(lambda b: b.get_size()).sum()
        
        k = int((2 * max(1.0, block_size / n_entities)))

        over_t_rdd = over_t.rdd.map(lambda x: x.asDict())

        def get_top_k(x):
            profile_id = x[0]
            neighbors_ids = sorted(x[1], key=lambda y: -y[1])

            if len(neighbors_ids) > k:
                neighbors_ids = neighbors_ids[:k]

            neighbors_ids = set(map(lambda y: y[0], neighbors_ids))

            return profile_id, neighbors_ids

        top_p1 = over_t_rdd.map(lambda x: (x['p1'], (x['p2'], x['p_match']))).groupByKey().map(get_top_k)
        top_p2 = over_t_rdd.map(lambda x: (x['p2'], (x['p1'], x['p_match']))).groupByKey().map(get_top_k)

        top_p1_brd = sc.broadcast(top_p1.collectAsMap())
        top_p2_brd = sc.broadcast(top_p2.collectAsMap())

        def do_pruning(p1, p2):
            return (p1 in top_p2_brd.value[p2]) and (p2 in top_p1_brd.value[p1])

        pruning_udf = f.udf(do_pruning, BooleanType())

        res = over_t \
            .select("p1", "p2", "p_match", "is_match", pruning_udf("p1", "p2").alias("keep")) \
            .where("keep") \
            .select("p1", "p2", "p_match", "is_match")
        res.count()

        over_t.unpersist()
        top_p1_brd.unpersist()
        top_p2_brd.unpersist()

        return res

    @staticmethod
    def cnp(edges, profiles=None, n_entities=-1, blocks=None, blocks_sum_sizes=-1):
        """
        Performs the pruning by using the Cardinality Node Pruning method
        :param edges: DataFrame of edges with the probability column (p_match)
        :param profiles: RDD of profiles
        :param blocks: RDD of blocks
        :return: dataframe with pruned edges
        """
        if blocks is None and blocks_sum_sizes < 0:
            raise Exception("Provide blocks or their sum of sizes")
        
        if profiles is None and n_entities < 0:
            raise Exception("Provide profiles or their number")
        
        sc = edges.rdd.context

        over_t = edges.filter("p_match >= 0.5")
        over_t.cache()
        over_t.count()

        if n_entities < 0:
            n_entities = profiles.count()
        
        if blocks_sum_sizes >= 0:
            block_size = blocks_sum_sizes
        else:
            block_size = blocks.map(lambda b: b.get_size()).sum()
            
        k = int((2 * max(1.0, block_size / n_entities)))

        over_t_rdd = over_t.rdd.map(lambda x: x.asDict())

        def get_top_k(x):
            profile_id = x[0]
            neighbors_ids = sorted(x[1], key=lambda y: -y[1])

            if len(neighbors_ids) > k:
                neighbors_ids = neighbors_ids[:k]

            neighbors_ids = set(map(lambda y: y[0], neighbors_ids))

            return profile_id, neighbors_ids

        top_p1 = over_t_rdd.map(lambda x: (x['p1'], (x['p2'], x['p_match']))).groupByKey().map(get_top_k)

        top_p2 = over_t_rdd.map(lambda x: (x['p2'], (x['p1'], x['p_match']))).groupByKey().map(get_top_k)

        top_p1_brd = sc.broadcast(top_p1.collectAsMap())
        top_p2_brd = sc.broadcast(top_p2.collectAsMap())

        def do_pruning(p1, p2):
            return (p1 in top_p2_brd.value[p2]) or (p2 in top_p1_brd.value[p1])

        pruning_udf = f.udf(do_pruning, BooleanType())

        res = over_t \
            .select("p1", "p2", "p_match", "is_match", pruning_udf("p1", "p2").alias("keep")) \
            .where("keep") \
            .select("p1", "p2", "p_match", "is_match")

        res.count()

        over_t.unpersist()
        top_p1_brd.unpersist()
        top_p2_brd.unpersist()

        return res

    @staticmethod
    def get_stats(edges, groundtruth=None, gt_size=0):
        """
        Given the dataframe of edges with the is_match column
        computes precision, recall and F1-score
        :param edges: dataframe of edges
        :param groundtruth: converted groundtruth
        :param gt_size: groundtruth size
        :return: precision, recall, F1-score
        """
        
        if groundtruth is None and gt_size == 0:
            raise Exception("You must provide the groundtruth or its size")
        
        if gt_size == 0:
            gt_size = len(groundtruth)

        num_matches = edges.where("is_match == 1").count()
        num_edges = edges.count()

        pc = num_matches / gt_size
        pq = num_matches / num_edges

        f1 = 0.0
        if pc > 0 and pq > 0:
            f1 = 2 * pc * pq / (pc + pq)

        return pc, pq, f1
