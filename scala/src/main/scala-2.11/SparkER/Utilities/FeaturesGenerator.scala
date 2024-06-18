package SparkER.Utilities

import SparkER.Utilities.SimilarityMeasures._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

object FeaturesGenerator {
  object AVAILABLE_FEATURES {
    val HAMMING_DIST = "HAMMING_DIST"
    val HAMMING_SIM = "HAMMING_SIM"
    val LEV_DIST = "LEV_DIST"
    val LEV_SIM = "LEV_SIM"
    val JARO = "JARO"
    val JARO_WINKLER = "JARO_WINKLER"
    val NEEDLEMAN_WUNSCH = "NEEDLEMAN_WUNSCH"
    val SMITH_WATERMAN = "SMITH_WATERMAN"
    val JACCARD = "JACCARD"
    val COSINE = "COSINE"
    val OVERLAP = "OVERLAP"
    val DICE = "DICE"
    val EXACT_MATCH = "EXACT_MATCH"
    val REL_DIFF = "REL_DIFF"
    val ABS_NORM = "ABS_NORM"

    val ALL_FEATURES = List(HAMMING_DIST, HAMMING_SIM, LEV_DIST, LEV_SIM, JARO, JARO_WINKLER, NEEDLEMAN_WUNSCH,
      SMITH_WATERMAN, JACCARD, COSINE, OVERLAP, DICE, EXACT_MATCH, REL_DIFF, ABS_NORM)

    val REQ_TOKENS = List(JACCARD, COSINE, OVERLAP, DICE)
  }

  def generateFeatures(df: DataFrame, mapping: List[(String, String)], features: List[String] = AVAILABLE_FEATURES.ALL_FEATURES, tokenizer: String => Array[String] = Splitter.tokens): DataFrame = {
    val transformers = mapping.flatMap { case (attr1, attr2) =>
      features.map { feature =>
        new GenerateFeatureTransformer().setInputCol1(attr1).setInputCol2(attr2).setFeature(feature).setTokenizer(tokenizer).setOutputCol(attr1 + "_" + attr2 + "_" + feature)
      }
    }

    /*val transformers = mapping.map { case (attr1, attr2) =>
      new GenerateFeaturesTransformer().setInputCol1(attr1).setInputCol2(attr2).setFeatures(features).setTokenizer(tokenizer)
    }*/

    val pipeline = new Pipeline().setStages(transformers.toArray)

    pipeline.fit(df).transform(df)
  }

  def getFeature(s1: String, s2: String, feature: String, tokenizer: String => Array[String]): Double = {
    feature match {
      case AVAILABLE_FEATURES.HAMMING_DIST => hammingDist(s1, s2).toDouble
      case AVAILABLE_FEATURES.HAMMING_SIM => hammingSim(s1, s2)
      case AVAILABLE_FEATURES.LEV_DIST => levDist(s1, s2).toDouble
      case AVAILABLE_FEATURES.LEV_SIM => levSim(s1, s2)
      case AVAILABLE_FEATURES.JARO => jaro(s1, s2)
      case AVAILABLE_FEATURES.JARO_WINKLER => jaroWinkler(s1, s2)
      case AVAILABLE_FEATURES.NEEDLEMAN_WUNSCH => needlemanWunsch(s1, s2)
      case AVAILABLE_FEATURES.SMITH_WATERMAN => smithWaterman(s1, s2)
      case AVAILABLE_FEATURES.JACCARD => jaccard(tokenizer(s1), tokenizer(s2))
      case AVAILABLE_FEATURES.COSINE => cosine(tokenizer(s1), tokenizer(s2))
      case AVAILABLE_FEATURES.OVERLAP => overlapCoeff(tokenizer(s1), tokenizer(s2))
      case AVAILABLE_FEATURES.DICE => dice(tokenizer(s1), tokenizer(s2))
      case AVAILABLE_FEATURES.EXACT_MATCH => exactMatch(s1, s2)
      case AVAILABLE_FEATURES.REL_DIFF => relDiff(s1, s2)
      case AVAILABLE_FEATURES.ABS_NORM => absNorm(s1, s2)
      case _ => Double.NaN
    }
  }

  def getFeature(s1: String, s2: String, feature: String, t1: Array[String], t2: Array[String]): Double = {
    feature match {
      case AVAILABLE_FEATURES.HAMMING_DIST => hammingDist(s1, s2).toDouble
      case AVAILABLE_FEATURES.HAMMING_SIM => hammingSim(s1, s2)
      case AVAILABLE_FEATURES.LEV_DIST => levDist(s1, s2).toDouble
      case AVAILABLE_FEATURES.LEV_SIM => levSim(s1, s2)
      case AVAILABLE_FEATURES.JARO => jaro(s1, s2)
      case AVAILABLE_FEATURES.JARO_WINKLER => jaroWinkler(s1, s2)
      case AVAILABLE_FEATURES.NEEDLEMAN_WUNSCH => needlemanWunsch(s1, s2)
      case AVAILABLE_FEATURES.SMITH_WATERMAN => smithWaterman(s1, s2)
      case AVAILABLE_FEATURES.JACCARD => jaccard(t1, t2)
      case AVAILABLE_FEATURES.COSINE => cosine(t1, t2)
      case AVAILABLE_FEATURES.OVERLAP => overlapCoeff(t1, t2)
      case AVAILABLE_FEATURES.DICE => dice(t1, t2)
      case AVAILABLE_FEATURES.EXACT_MATCH => exactMatch(s1, s2)
      case AVAILABLE_FEATURES.REL_DIFF => relDiff(s1, s2)
      case AVAILABLE_FEATURES.ABS_NORM => absNorm(s1, s2)
      case _ => Double.NaN
    }
  }
}
