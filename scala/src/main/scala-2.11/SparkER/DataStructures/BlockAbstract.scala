package SparkER.DataStructures

/**
  * Represents a generic block.
  *
  * @author Giovanni Simonini
  * @since 2016/12/07
  */
trait BlockAbstract extends Ordered[BlockAbstract] {
  val blockingKey: String
  /** Id of the block */
  val blockID: Long
  /** Entropy of the block */
  var entropy: Double
  /** Cluster */
  var clusterID: Integer
  /** Id of the profiles contained in the block */
  val profiles: Array[Set[Long]]

  /** Return the number of entities indexed in the block */
  def size: Double = profiles.map(_.size.toDouble).sum

  /* Return the number of comparisons entailed by this block */
  def getComparisonSize(): Double

  /* Returns all profiles */
  def getAllProfiles: Array[Long] = profiles.flatten

  /* Returns all the comparisons */
  def getComparisons(): Set[(Long, Long)]

  /** Returns the number of comparisons entailed by this profile ID */
  def getComparisonSize(profileID: Long): Double

  /** Default comparator, blocks will be ordered by its comparison size */
  def compare(that: BlockAbstract): Int = {
    this.getComparisonSize() compare that.getComparisonSize()
  }
}