package SparkER.EntityMatching

import SparkER.DataStructures.{Profile, UnweightedEdge, WeightedEdge}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

object EntityMatcher {

  ///todo: tokenizzare prima di mandare in broadcast, è meglio lavorare già con (profileID -> lista token)
  def entityMatching(profilesBroadcast: Broadcast[scala.collection.Map[Long, Profile]], candidatePairs: RDD[UnweightedEdge], threshold: Double,
                     matchingFunction: (Profile, Profile) => Double = MatchingFunctions.jaccardSimilarity)
  : RDD[WeightedEdge] = {
    val scored = candidatePairs.map { pair =>
      //if (profilesBroadcast.value.contains(pair.firstProfileID) && profilesBroadcast.value.contains(pair.secondProfileID)) {
      val similarity = matchingFunction(profilesBroadcast.value(pair.firstProfileID), profilesBroadcast.value(pair.secondProfileID))
      WeightedEdge(pair.firstProfileID, pair.secondProfileID, similarity)
      //}
    }

    scored.filter(_.weight >= threshold)
  }

}
