package Experiments

import java.util.Calendar

import SparkER.BlockBuildingMethods.TokenBlocking
import SparkER.BlockRefinementMethods.PruningMethods._
import SparkER.BlockRefinementMethods.{BlockFiltering, BlockPurging}
import SparkER.Utilities.Converters
import SparkER.Wrappers.{CSVWrapper, JSONWrapper}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.parsing.json.JSON

import SparkER.DataStructures.Profile
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.count
import org.apache.hadoop.fs._

/**
  * Experiments
  *
  * @author Luca Gagliardelli
  * @since 18/12/2018
  **/
object CalcCBS {

  case class Dataset(name: String, basePath: String, dataset1: String, dataset2: String, groundtruth: String, format: String, dtype: String, gt_d1_field: String, gt_d2_field: String, purgingThreshold: Double = 1.0, idField: String = "realProfileID")

  def calcCBS(dataset: Dataset, outputPath: String) = {

    val sc = SparkContext.getOrCreate()
    val sparkSession = SparkSession.builder().getOrCreate()

    //Carico i dati
    val dataset1 = {
      if (dataset.format == "json") {
        JSONWrapper.loadProfiles(dataset.basePath + dataset.dataset1, realIDField = dataset.idField, sourceId = 1)
      }
      else {
        CSVWrapper.loadProfiles2(dataset.basePath + dataset.dataset1, realIDField = dataset.idField, sourceId = 1, header = true)
      }
    }

    //Separatore
    val maxIdDataset1 = dataset1.map(_.id).max()

    val dataset2: RDD[Profile] = {
      if (dataset.dtype == "clean") {
        if (dataset.format == "json") {
          JSONWrapper.loadProfiles(dataset.basePath + dataset.dataset2, realIDField = dataset.idField, sourceId = 2, startIDFrom = maxIdDataset1 + 1)
        }
        else {
          CSVWrapper.loadProfiles2(dataset.basePath + dataset.dataset2, realIDField = dataset.idField, sourceId = 2, startIDFrom = maxIdDataset1 + 1, header = true)
        }
      }
      else {
        null
      }
    }

    //ID massimo dei profili
    val maxID = {
      if (dataset.dtype == "clean") {
        dataset2.map(_.id).max().toInt
      }
      else {
        maxIdDataset1.toInt
      }
    }

    //Definisco i separatori
    val separators = {
      if (dataset.dtype == "clean") {
        Array(maxIdDataset1)
      }
      else {
        Array.emptyLongArray
      }
    }

    val profiles = {
      if (dataset.dtype == "clean") {
        dataset1.union(dataset2)
      }
      else {
        dataset1
      }
    }

    //Carico il groundtruth
    val groundtruth = {
      if (dataset.format == "json") {
        JSONWrapper.loadGroundtruth(dataset.basePath + dataset.groundtruth, firstDatasetAttribute = dataset.gt_d1_field, secondDatasetAttribute = dataset.gt_d2_field)
      }
      else {
        CSVWrapper.loadGroundtruth(dataset.basePath + dataset.groundtruth)
      }
    }

    val t1 = Calendar.getInstance()

    //Token blocking
    val blocks = TokenBlocking.createBlocks(profiles, separators)

    val blocksPurged = BlockPurging.blockPurging(blocks, dataset.purgingThreshold)

    val profileBlocks = Converters.blocksToProfileBlocks(blocksPurged)
    val profileBlocksFiltered = BlockFiltering.blockFiltering(profileBlocks, 0.8)
    val blocksAfterFiltering = Converters.profilesBlockToBlocks(profileBlocksFiltered, separators)

    //Invio in broadcast i blocchi
    val blockIndexMap = blocksAfterFiltering.map(b => (b.blockID, b.profiles)).collectAsMap()
    val blockIndex = sc.broadcast(blockIndexMap)

    //Questo serve per riconvertire le coppie con gli id originali
    val profilesIds = sc.broadcast(profiles.map(p => (p.id, p.originalID)).collectAsMap())

    //For each partition
    val pairsFeatures = profileBlocksFiltered.mapPartitions { part =>
      val features: ListBuffer[(String, String, Double)] = ListBuffer[(String, String, Double)]()

      val CBS = Array.fill[Double](maxID + 1) {
        0
      }
      val neighbors = Array.ofDim[Int](maxID + 1)
      var neighborsNumber = 0

      //For each profile
      part.foreach { pb =>
        val profileID = pb.profileID.toInt

        //For each block
        pb.blocks.foreach { block =>
          //Block ID
          val blockID = block.blockID
          //Gets the block data from the index
          val blockData = blockIndex.value.get(blockID)
          if (blockData.isDefined) {
            //Gets all the neighbors, i.e. other profiles in the block
            PruningUtils.getAllNeighbors(profileID, blockData.get, separators).foreach { neighbourID =>
              if (profileID < neighbourID) {
                CBS.update(neighbourID.toInt, CBS(neighbourID.toInt) + 1)
                if (CBS(neighbourID.toInt) == 1) {
                  neighbors.update(neighborsNumber, neighbourID.toInt)
                  neighborsNumber += 1
                }
              }
            }
          }
        }

        for (i <- 0 until neighborsNumber) {
          //Emetto la feature per questa coppia: ID PROFILO 1, ID PROFILO 2, CFIBF, RACCB, JS, Numero di comparison non ridondanti di P1, Numero di comparison non ridondanti di P2, Match o meno
          features.append((profilesIds.value(profileID), profilesIds.value(neighbors(i)), CBS(neighbors(i))))
          CBS.update(neighbors(i), 0)
        }
        neighborsNumber = 0
      }

      features.toIterator
    }

    val featuresDF = sparkSession.createDataFrame(pairsFeatures).toDF("p1", "p2", "CBS")
    val gtDF = sparkSession.createDataFrame(groundtruth.map(g => (g.firstEntityID, g.secondEntityID))).toDF("gt1", "gt2")

    val res = gtDF.join(featuresDF, (col("gt1") === col("p1")) and (col("p2") === col("gt2")), "left")

    val res2 = res.na.fill(0)

    res2.groupBy("CBS").agg(count("*").alias("cnt")).coalesce(1).write.option("header", "True").csv(outputPath + dataset.name)

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val file = fs.globStatus(new Path(outputPath + dataset.name + "/part*"))(0).getPath().getName()

    fs.rename(new Path(outputPath + dataset.name + "/" + file), new Path(outputPath + dataset.name + ".csv"))
    fs.delete(new Path(outputPath + dataset.name), true)

    profilesIds.unpersist()
    blockIndex.unpersist()
  }


  def main(args: Array[String]): Unit = {
    val config = Source.fromFile("/home/app/config/config.ini")
    val spark_max_memory = config.getLines().filter(_.startsWith("max_memory=")).next().replace("max_memory=", "")
    config.close()

    val input = Source.fromFile("/home/app/datasets/datasets.json")
    val lines = input.mkString
    input.close()
    val json = JSON.parseFull(lines)

    val datasets = json.get.asInstanceOf[List[Map[String, String]]].map(d => {
      val name = d("name")
      val basePath = "/home/app/" + d("base_path") + "/"
      val dataset1 = d("d1")
      val dataset2 = d.getOrElse("d2", "")
      val groundtruth = d("gt")
      val format = d("format")
      val dtype = d("type")
      val idField = d("d1_id_field")
      val gt_d1_field = d("gt_d1_field")
      val gt_d2_field = d("gt_d2_field")
      val purging_threshold = d.getOrElse("purging_threshold", "1.0").toDouble

      Dataset(name, basePath, dataset1, dataset2, groundtruth, format, dtype, gt_d1_field, gt_d2_field, idField = idField, purgingThreshold = purging_threshold)
    })

    val conf = new SparkConf()
      .setAppName("Main")
      .setMaster("local[*]")
      .set("spark.driver.memory", spark_max_memory)
      .set("spark.executor.memory", spark_max_memory)
      .set("spark.local.dir", "/home/app/tmp")
      .set("spark.driver.maxResultSize", "0")

    val outputPath = "/home/app/cbs_stats/"

    val sc = new SparkContext(conf)

    datasets.filter(d => d.dtype == "clean").foreach { d =>
      calcCBS(d, outputPath)
    }

    sc.stop()
  }
}
