package org.apache.spark.ml

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, _}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * Created by gaurav.prasad on 17/11/17.
  */
class StringMerge(override val uid: String) extends Transformer with DefaultParamsWritable {
  final val inputCol1: Param[String] = new Param[String](this, "inputCol1", "input first column name")
  final def getInputCol1: String = $(inputCol1)

  final val inputCol2: Param[String] = new Param[String](this, "inputCol2", "input second column name")
  final def getInputCol2: String = $(inputCol2)

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final def getOutputCol: String = $(outputCol)

  def setInputCol1(value: String): this.type = set(inputCol1, value)
  def setInputCol2(value: String): this.type = set(inputCol2, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val mergeAddress = udf((s1: String, s2: String) => (s1 + " " + s2).trim)

  def this() = this(Identifiable.randomUID("StringMerge"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    dataset
      .withColumn(getOutputCol, mergeAddress(col(getInputCol1), col(getInputCol2)))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields :+ StructField(getOutputCol, StringType))
  }
}

object StringMerge extends DefaultParamsReadable[StringMerge]
