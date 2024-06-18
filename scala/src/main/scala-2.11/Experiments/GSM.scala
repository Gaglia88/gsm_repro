package Experiments

object GSM {

  case class DatasetStats(name: String, blockSize: Int, nOfEntities: Int)

  def calcPcPqF1Blast(df: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {

    val sc = df.sparkSession.sparkContext

    val initialMatches = sc.doubleAccumulator

    val w1 = df.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val proba = x.getAs[Double]("p_match")
      val is_match = x.getAs[Int]("is_match")

      if (is_match == 1) {
        initialMatches.add(1)
      }

      val p = proba.toFloat
      (p1, p)
    }.filter(_._2 >= 0.5)

    val w2 = df.rdd.map { x =>
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      val p = proba.toFloat
      (p2, p)
    }.filter(_._2 >= 0.5)

    val maxv1 = w1.reduceByKey((x, y) => math.max(x, y))
    val maxv2 = w2.reduceByKey((x, y) => math.max(x, y))
    val profileMaxProba1 = sc.broadcast(maxv1.collectAsMap())
    val profileMaxProba2 = sc.broadcast(maxv2.collectAsMap())

    val finalComparisons = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator

    df.rdd.foreach { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match").toFloat

      if (proba >= 0.5) {
        val threshold = 0.35 * (profileMaxProba1.value(p1) + profileMaxProba2.value(p2))
        if (proba >= threshold) {
          val is_match = x.getAs[Int]("is_match")
          if (is_match == 1) {
            finalMatches.add(1)
          }
          finalComparisons.add(1)
        }
      }
    }

    val pc = finalMatches.value / initialMatches.value
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1RCNP(df: org.apache.spark.sql.DataFrame = null, dataset: DatasetStats): (Double, Double, Double) = {
    val k = (2 * math.max(1.0, dataset.blockSize.toFloat / dataset.nOfEntities.toFloat)).toInt

    val sc = df.sparkSession.sparkContext

    val initialMatches = df.filter(df("is_match") === 1).count()

    val overT = df.filter(df("p_match") >= 0.5)

    val topP1 = overT.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      (p1, (p2, proba))
    }.groupByKey().map { x =>
      val topEl = x._2.toArray.sortBy(-_._2).take(k).map(_._1).toSet
      (x._1, topEl)
    }

    val topP2 = overT.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      (p2, (p1, proba))
    }.groupByKey().map { x =>
      val topEl = x._2.toArray.sortBy(-_._2).take(k).map(_._1).toSet
      (x._1, topEl)
    }

    val profileTopP1 = sc.broadcast(topP1.collectAsMap())
    val profileTopP2 = sc.broadcast(topP2.collectAsMap())

    val finalComparisons = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator

    overT.rdd.foreach { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")

      if (profileTopP1.value(p1).contains(p2) && profileTopP2.value(p2).contains(p1)) {
        val is_match = x.getAs[Int]("is_match")
        if (is_match == 1) {
          finalMatches.add(1)
        }
        finalComparisons.add(1)
      }
    }

    val pc = finalMatches.value / initialMatches
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1CNP(df: org.apache.spark.sql.DataFrame = null, dataset: DatasetStats): (Double, Double, Double) = {
    val k = (2 * math.max(1.0, dataset.blockSize.toFloat / dataset.nOfEntities.toFloat)).toInt

    val sc = df.sparkSession.sparkContext

    val initialMatches = df.filter(df("is_match") === 1).count()

    val overT = df.filter(df("p_match") >= 0.5)

    val topP1 = overT.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      (p1, (p2, proba))
    }.groupByKey().map { x =>
      val topEl = x._2.toArray.sortBy(-_._2).take(k).map(_._1).toSet
      (x._1, topEl)
    }

    val topP2 = overT.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      (p2, (p1, proba))
    }.groupByKey().map { x =>
      val topEl = x._2.toArray.sortBy(-_._2).take(k).map(_._1).toSet
      (x._1, topEl)
    }

    val profileTopP1 = sc.broadcast(topP1.collectAsMap())
    val profileTopP2 = sc.broadcast(topP2.collectAsMap())

    val finalComparisons = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator

    overT.rdd.foreach { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")

      if (profileTopP1.value(p1).contains(p2) || profileTopP2.value(p2).contains(p1)) {
        val is_match = x.getAs[Int]("is_match")
        if (is_match == 1) {
          finalMatches.add(1)
        }
        finalComparisons.add(1)
      }
    }

    val pc = finalMatches.value / initialMatches
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1WNP(df: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {
    val sc = df.sparkSession.sparkContext

    val initialMatches = sc.doubleAccumulator

    val w1 = df.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val proba = x.getAs[Double]("p_match")
      val is_match = x.getAs[Int]("is_match")

      if (is_match == 1) {
        initialMatches.add(1)
      }

      val p = proba.toFloat
      (p1, p)
    }.filter(_._2 >= 0.5)
    val w2 = df.rdd.map { x =>
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      val p = proba.toFloat
      (p2, p)
    }.filter(_._2 >= 0.5)

    val thresholdP1 = sc.broadcast(w1.groupByKey().map(x => (x._1, x._2.sum / x._2.size)).collectAsMap())
    val thresholdP2 = sc.broadcast(w2.groupByKey().map(x => (x._1, x._2.sum / x._2.size)).collectAsMap())

    val finalComparisons = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator

    df.rdd.foreach { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match").toFloat

      if (proba >= 0.5) {
        if (proba >= thresholdP1.value(p1) || proba >= thresholdP2.value(p2)) {
          val is_match = x.getAs[Int]("is_match")
          if (is_match == 1) {
            finalMatches.add(1)
          }
          finalComparisons.add(1)
        }
      }
    }

    val pc = finalMatches.value / initialMatches.value
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1RWNP(df: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {
    val sc = df.sparkSession.sparkContext

    val initialMatches = sc.doubleAccumulator

    val w1 = df.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val proba = x.getAs[Double]("p_match")
      val is_match = x.getAs[Int]("is_match")

      if (is_match == 1) {
        initialMatches.add(1)
      }

      val p = proba.toFloat
      (p1, p)
    }.filter(_._2 >= 0.5)
    val w2 = df.rdd.map { x =>
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match")
      val p = proba.toFloat
      (p2, p)
    }.filter(_._2 >= 0.5)

    val thresholdP1 = sc.broadcast(w1.groupByKey().map(x => (x._1, x._2.sum / x._2.size)).collectAsMap())
    val thresholdP2 = sc.broadcast(w2.groupByKey().map(x => (x._1, x._2.sum / x._2.size)).collectAsMap())

    val finalComparisons = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator

    df.rdd.foreach { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      val proba = x.getAs[Double]("p_match").toFloat

      if (proba >= 0.5) {
        if (proba >= thresholdP1.value(p1) && proba >= thresholdP2.value(p2)) {
          val is_match = x.getAs[Int]("is_match")
          if (is_match == 1) {
            finalMatches.add(1)
          }
          finalComparisons.add(1)
        }
      }
    }

    thresholdP1.unpersist()
    thresholdP2.unpersist()

    val pc = finalMatches.value / initialMatches.value
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1Bcl(df: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {
    val sc = df.sparkSession.sparkContext
    val initialMatches = sc.doubleAccumulator
    val finalMatches = sc.doubleAccumulator
    val finalComparisons = sc.doubleAccumulator

    df.rdd.foreach { x =>
      val pred = x.getAs[Int]("pred")
      val is_match = x.getAs[Int]("is_match")

      if (is_match == 1) {
        initialMatches.add(1)
      }

      if (pred == 1) {
        finalComparisons.add(1)
        if (is_match == 1) {
          finalMatches.add(1)
        }
      }
    }

    val pc = finalMatches.value / initialMatches.value
    val pq = finalMatches.value / finalComparisons.value
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1CEP(df: org.apache.spark.sql.DataFrame, dataset: DatasetStats): (Double, Double, Double) = {
    val sc = df.sparkSession.sparkContext
    val number_of_edges_to_keep = dataset.blockSize / 2

    val initialMatches = sc.doubleAccumulator

    val data = df.rdd.map { x =>
      val is_match = x.getAs[Int]("is_match")
      val proba = x.getAs[Double]("p_match")
      if (is_match == 1) {
        initialMatches.add(1)
      }
      (proba, is_match)
    }.filter(_._1 >= 0.5)

    val to_keep = data.takeOrdered(number_of_edges_to_keep)(Ordering[Double].reverse.on(x => x._1))

    val finalMatches = to_keep.map(_._2).sum
    val finalComparisons = to_keep.length

    val pc = finalMatches / initialMatches.value
    val pq = finalMatches / finalComparisons.toDouble
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }

  def calcPcPqF1WEP(df: org.apache.spark.sql.DataFrame): (Double, Double, Double) = {

    val initialMatches = df.where("is_match == 1").count()
    val valid = df.where("p_match >= 0.5")
    val threshold = valid.rdd.map(x => x.getAs[Double]("p_match")).mean()
    val filtered = valid.where("p_match >= "+threshold)
    val finalComparisons = filtered.count()
    val finalMatches = filtered.where("is_match == 1").count().toDouble

    val pc = finalMatches / initialMatches
    val pq = finalMatches / finalComparisons
    val f1 = 2 * pc * pq / (pc + pq)
    (pc, pq, f1)
  }
}
