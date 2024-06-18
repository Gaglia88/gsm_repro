package SparkER.Utilities

import com.github.vickumar1981.stringdistance.StringDistance._
import com.github.vickumar1981.stringdistance.impl.{ConstantGap, LinearGap}

object SimilarityMeasures {

  def hammingDist(s1: String, s2: String): Int = {
    Hamming.distance(s1, s2)
  }

  def hammingSim(s1: String, s2: String): Double = {
    Hamming.score(s1, s2)
  }

  def levDist[A](a: Iterable[A], b: Iterable[A]): Int = {
    ((0 to b.size).toList /: a) ((prev, x) =>
      (prev zip prev.tail zip b).scanLeft(prev.head + 1) {
        case (h, ((d, v), y)) => math.min(math.min(h + 1, v + 1), d + (if (x == y) 0 else 1))
      }).last
  }

  def levSim(s1: String, s2: String): Double = {
    levDist(s1, s2).toDouble / math.max(s1.length, s2.length)
  }

  def jaro(s1: String, s2: String): Double = {
    Jaro.score(s1, s2)
  }

  def jaroWinkler(s1: String, s2: String): Double = {
    JaroWinkler.score(s1, s2, 0.1)
  }

  def needlemanWunsch(s1: String, s2: String): Double = {
    NeedlemanWunsch.score(s1, s2, ConstantGap())
  }

  def smithWaterman(s1: String, s2: String): Double = {
    SmithWaterman.score(s1, s2, (LinearGap(gapValue = -1), Integer.MAX_VALUE))
  }

  def jaccard(arr1: Array[String], arr2: Array[String]): Double = {
    val common = arr1.intersect(arr2).length.toDouble
    common / (arr1.length + arr2.length - common)
  }

  def cosine(arr1: Array[String], arr2: Array[String]): Double = {
    val el = arr1.union(arr2).distinct
    val a: Array[Double] = Array.fill[Double](el.length)(0)
    val b: Array[Double] = Array.fill[Double](el.length)(0)

    for (i <- el.indices) {
      if (arr1.contains(el(i))) {
        a.update(i, 1)
      }
      if (arr2.contains(el(i))) {
        b.update(i, 1)
      }
    }

    val dot = a.zip(b).map(x => x._1 * x._2).sum

    val normA = math.sqrt(a.sum)
    val normB = math.sqrt(b.sum)

    dot / (normA * normB)
  }

  def overlapCoeff(arr1: Array[String], arr2: Array[String]): Double = {
    arr1.intersect(arr2).length
  }

  def dice(arr1: Array[String], arr2: Array[String]): Double = {
    2 * arr1.intersect(arr2).length.toDouble / (arr1.length.toDouble + arr2.length)
  }

  def exactMatch(d1: Any, d2: Any): Double = {
    if (d1 == null || d2 == null) {
      Double.NaN
    }
    else {
      if (d1 == d2) {
        1.0
      }
      else {
        0.0
      }
    }
  }

  def relDiff(d1: Any, d2: Any): Double = {
    if (d1 == null || d2 == null) {
      Double.NaN
    }
    else {
      try {
        val v1 = d1.toString.toDouble
        val v2 = d2.toString.toDouble
        if (v1 == 0.0 && v2 == 0.0) {
          0
        }
        else {
          (2 * math.abs(v1 - v2)) / (v1 + v2)
        }
      }
      catch {
        case _ =>
          Double.NaN
      }
    }
  }

  def absNorm(d1: Any, d2: Any): Double = {
    if (d1 == null || d2 == null) {
      Double.NaN
    }
    else {
      try {
        val v1 = d1.toString.toDouble
        val v2 = d2.toString.toDouble
        if (v1 == 0.0 && v2 == 0.0) {
          0
        }
        else {
          var x = (math.abs(v1 - v2) / math.max(math.abs(v1), math.abs(v2)))
          if (x <= 10e-5) {
            x = 0
          }
          1.0 - x
        }
      }
      catch {
        case _ =>
          Double.NaN
      }
    }
  }

}
