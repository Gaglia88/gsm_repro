package Experiments

import java.io.{FileWriter, PrintWriter}

import Experiments.GSM.DatasetStats
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.io.Source

object ThresholdCalc {
  def calcThresholdsBlast(df: org.apache.spark.sql.DataFrame): (Double, Double, Double, Double) = {

    val sc = df.sparkSession.sparkContext

    val w1 = df.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      var p: Float = 0
      try {
        p = x.getAs[Double]("p_match").toFloat
      }
      catch {
        case _: ClassCastException =>
          p = x.getAs[Float]("p_match")
      }

      (p1, p)
    }.filter(_._2 >= 0.5)

    val w2 = df.rdd.map { x =>
      val p2 = x.getAs[String]("p2")
      var p: Float = 0
      try {
        p = x.getAs[Double]("p_match").toFloat
      }
      catch {
        case _: ClassCastException =>
          p = x.getAs[Float]("p_match")
      }
      (p2, p)
    }.filter(_._2 >= 0.5)

    val maxv1 = w1.reduceByKey((x, y) => math.max(x, y))
    val maxv2 = w2.reduceByKey((x, y) => math.max(x, y))
    val profileMaxProba1 = sc.broadcast(maxv1.collectAsMap())
    val profileMaxProba2 = sc.broadcast(maxv2.collectAsMap())

    val finalComparisons = sc.doubleAccumulator

    val thresholds = df.rdd.map { x =>
      val p1 = x.getAs[String]("p1")
      val p2 = x.getAs[String]("p2")
      var res: Double = -1
      var p: Float = 0
      try {
        p = x.getAs[Double]("p_match").toFloat
      }
      catch {
        case _: ClassCastException =>
          p = x.getAs[Float]("p_match")
      }

      if (p >= 0.5) {
        val threshold = 0.35 * (profileMaxProba1.value(p1) + profileMaxProba2.value(p2))
        if (p >= threshold) {
          finalComparisons.add(1)
        }
        res = threshold
      }
      res
    }.filter(_ >= 0)

    val max = thresholds.max()
    val min = thresholds.min()
    val avg = thresholds.mean()

    (min, max, avg, finalComparisons.value)
  }

  def main(args: Array[String]): Unit = {
    val config = Source.fromFile("/home/app/config/config.ini")
    val lines = config.getLines().toList
    val rep = lines.filter(_.startsWith("repetitions=")).head.replace("repetitions=", "").toInt
    val spark_max_memory = lines.filter(_.startsWith("max_memory=")).head.replace("max_memory=", "")
    config.close()

    val input = Source.fromFile("/home/app/results/01b_blocking_stats.csv")
    val datasets = input.getLines().drop(1).map(l => {
      val data = l.split(",")
      data(0)
      DatasetStats(data(0), data(2).toDouble.toInt, data(1).toInt)
    })
    input.close()

    var processed = ""
    try {
      val input2 = Source.fromFile("/home/app/results/blast_thresholds.csv")
      processed = input2.mkString
      input2.close()
    }
    catch {
      case _: Throwable => processed = ""
    }

    val conf = new SparkConf()
      .setAppName("Main")
      .setMaster("local[*]")
      .set("spark.driver.memory", spark_max_memory)
      .set("spark.executor.memory", spark_max_memory)
      .set("spark.local.dir", "/home/app/tmp")
      .set("spark.driver.maxResultSize", "0")

    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder().getOrCreate()

    val train_set_sizes = List(20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    val features_set_id = List(78)


    val out = new PrintWriter(new FileWriter("/home/app/results/blast_thresholds.csv", true))
    if (processed.length == 0) {
      out.println("dataset;feature_set_id;train_size;algorithm;min;max;avg;comp_num")
    }

    datasets.foreach(d => {
      if (d.name == "AbtBuy") {
        features_set_id.foreach(feature_set_id => {
          train_set_sizes.foreach(train_set_size => {
            if (!processed.contains(d.name + ";" + train_set_size + ";" + feature_set_id)) {

              val algorithms = feature_set_id match {
                case 78 => List("blast")
                case 128 => List("bcl")
                case 187 => List("rcnp")
              }

              val results = collection.mutable.Map[String, collection.mutable.Map[String, List[Double]]]()

              algorithms.foreach(a => {
                val amap = collection.mutable.Map[String, List[Double]]()
                amap("min") = Nil
                amap("max") = Nil
                amap("avg") = Nil
                amap("comp") = Nil
                results(a) = amap
              })

              for (run <- 0 until rep) {
                val dataPath = "/home/app/probabilities/" + d.name + "/" + train_set_size + "/" + d.name + "_fs" + feature_set_id + "_run" + run + ".parquet"
                val df = sparkSession.read.parquet(dataPath)
                df.cache()
                df.count()

                algorithms.foreach(a => {
                  val (min, max, avg, comp) = a match {
                    case "blast" => calcThresholdsBlast(df)
                  }

                  results(a)("min") = min :: results(a)("min")
                  results(a)("max") = max :: results(a)("max")
                  results(a)("avg") = avg :: results(a)("avg")
                  results(a)("comp") = comp :: results(a)("comp")
                })
                df.unpersist()
              }

              algorithms.foreach(a => {
                val min = results(a)("min").sum / results(a)("min").length
                val max = results(a)("max").sum / results(a)("max").length
                val avg = results(a)("avg").sum / results(a)("avg").length
                val comp = results(a)("comp").sum / results(a)("comp").length

                out.println(d.name + ";" + feature_set_id + ";" + train_set_size + ";" + a + ";" + min + ";" + max + ";" + avg + ";" + comp)
              })

              out.flush()
            }
          })
        })
      }
    })

    out.close()
    sc.stop()
  }
}
