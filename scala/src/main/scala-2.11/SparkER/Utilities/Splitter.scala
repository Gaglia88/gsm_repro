package SparkER.Utilities

object Splitter {
  def bigrams(s: String): Array[String] = {
    s.sliding(2).toArray
  }

  def trigrams(s: String): Array[String] = {
    s.sliding(3).toArray
  }

  def tokens(s: String): Array[String] = {
    s.split("[\\W_]")
  }
}
