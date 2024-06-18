package SparkER.EntityMatching

import SparkER.DataStructures.Profile

object MatchingFunctions {

  def getTokens(p: Profile): Set[String] = {
    p.attributes.map(_.value).flatMap(_.split(SparkER.BlockBuildingMethods.BlockingUtils.TokenizerPattern.DEFAULT_SPLITTING)).map(_.toLowerCase.trim).filter(_.length > 0).toSet
  }

  def jaccardSimilarity(p1: Profile, p2: Profile): Double = {
    val t1 = getTokens(p1)
    val t2 = getTokens(p2)
    val common = t1.intersect(t2).size.toDouble
    common / (t1.size + t2.size - common)
  }

}
