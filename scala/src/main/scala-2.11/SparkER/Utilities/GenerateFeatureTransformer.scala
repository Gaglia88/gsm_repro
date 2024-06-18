package SparkER.Utilities

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DataTypes, StructType}

class GenerateFeatureTransformer(val uid: String = "1234") extends Transformer {

  /** Set the first input column name */
  final val inputCol1: Param[String] = new Param[String](this, "inputCol1", "input column name")

  final def getInputCol1: String = $(inputCol1)

  final def setInputCol1(value: String): GenerateFeatureTransformer = set(inputCol1, value)

  /** Set the second input column name */
  final val inputCol2: Param[String] = new Param[String](this, "inputCol2", "input column name")

  final def getInputCol2: String = $(inputCol2)

  final def setInputCol2(value: String): GenerateFeatureTransformer = set(inputCol2, value)

  /** Set the feature to compute */
  final val feature: Param[String] = new Param[String](this, "feature", "feature to compute")

  final def getFeature: String = $(feature)

  final def setFeature(value: String): GenerateFeatureTransformer = set(feature, value)

  /** Set the tokenizer to use */
  final val tokenizer: Param[String => Array[String]] = new Param[String => Array[String]](this, "tokenizer", "tokenizer to employ")

  final def getTokenizer: String => Array[String] = $(tokenizer)

  final def setTokenizer(value: String => Array[String]): GenerateFeatureTransformer = set(tokenizer, value)

  /** Name of the output column */
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  final def getOutputCol: String = $(outputCol)

  final def setOutputCol(value: String): GenerateFeatureTransformer = set(outputCol, value)

  private def calcFeature(s1: String, s2: String): Double = {
    if(s1 == null || s2 == null){
      Double.NaN
    }
    else{
      FeaturesGenerator.getFeature(s1, s2, $(feature), $(tokenizer))
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val calcFeatureUDF = udf(calcFeature _)
    dataset.withColumn($(outputCol), calcFeatureUDF(col($(inputCol1)), col($(inputCol2))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // Validate input type.
    // Input type validation is technically optional, but it is a good practice since it catches
    // schema errors early on.
    val actualDataType1 = schema($(inputCol1)).dataType
    val actualDataType2 = schema($(inputCol1)).dataType

    require(actualDataType1.equals(DataTypes.StringType), s"Column ${$(inputCol1)} must be StringType but was actually $actualDataType1.")
    require(actualDataType2.equals(DataTypes.StringType), s"Column ${$(inputCol1)} must be StringType but was actually $actualDataType2.")

    // Compute output type.
    // This is important to do correctly when plugging this Transformer into a Pipeline,
    // where downstream Pipeline stages may expect use this Transformer's output as their input.

    DataTypes.createStructType(
      (schema.fields.toList ::: List(DataTypes.createStructField($(outputCol), DataTypes.DoubleType, false))).toArray
    )
  }
}
