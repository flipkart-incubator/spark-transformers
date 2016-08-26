package com.flipkart.fdp.ml

import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.param.{ParamMap, Param, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}

/**
 * Created by shubhranshu.shekhar on 17/07/16.
 */
trait ProbabilityTransformParams extends Params {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  setDefault(outputCol, uid + "__output")
  final def getOutputCol: String = $(outputCol)

  final val actualClickProportion: Param[Double] = new Param[Double](this, "actualProportion", "click proportion actual")
  final def getActualClickProportion: Double = $(actualClickProportion)

  final val underSampledClickProportion: Param[Double] = new Param[Double](this, "undersampledProportion", "click proportion under-sampled")
  final def getUnderSampledClickProportion: Double = $(underSampledClickProportion)

  final val probIndex: Param[Int] = new Param[Int](this, "probIndex", "Index of prob for Vector type column")
  setDefault(probIndex, 1)
  final def getProbIndex: Int = $(probIndex)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    //val typeCandidates = List(new ArrayType(DoubleType, true), new ArrayType(StringType, false))
    //CustomSchemaUtil.checkColumnTypes(schema, $(inputCol), typeCandidates)
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    CustomSchemaUtil.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
 * ProbabilityTransform: Removing the effects of under-sampling, and yields true probability.
 */
class ProbabilityTransform(override val uid: String) extends Estimator[ProbabilityTransformModel]
  with ProbabilityTransformParams{
  def this() = this(Identifiable.randomUID("probabilityTransform"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setActualClickProportion(value: Double): this.type = set(actualClickProportion, value)

  def setUnderSampledClickProportion(value: Double): this.type = set(underSampledClickProportion, value)

  def setProbIndex(value: Int): this.type = set(probIndex, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def fit(dataset: DataFrame): ProbabilityTransformModel = {
    transformSchema(dataset.schema)
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val p1 = $(actualClickProportion)
    val r1 = $(underSampledClickProportion)
    require(p1 > 0 && p1 < 1, "Actual positive class proportion must be between 0 and 1")
    require(r1 > 0 && r1 < 1, "Under-sampled positive class proportion must be between 0 and 1")
    require(p1 < r1, "After under-sampling proportion of positive class should be greater compared to actual proportion.")
    //testing for a Vector column
    val dataType = new VectorUDT
    val inputSchema = dataset.schema
    val actualDataType = inputSchema(inputColName).dataType

    if(dataType.equals(actualDataType)){
      copyValues(new ProbabilityTransformModel(uid, p1, r1, $(probIndex)).setInputCol(inputColName).setOutputCol(outputColName).setParent(this))
    }
    else{
      copyValues(new ProbabilityTransformModel(uid, p1, r1, -1).setInputCol(inputColName).setOutputCol(outputColName).setParent(this))
    }

  }
  override def copy(extra: ParamMap): ProbabilityTransform = defaultCopy(extra)

}

class ProbabilityTransformModel(override val uid: String, val actualProportionOfClicks: Double,
                                val underSampledProportionOfClicks: Double, probFieldIndex: Int)
  extends Model[ProbabilityTransformModel] with ProbabilityTransformParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val p1 = actualProportionOfClicks
    val r1 = underSampledProportionOfClicks
    val i = probFieldIndex

    val scale = if(i < 0) {
      udf{ ppctr: Double =>
        (ppctr *p1/r1) / ((ppctr *p1/r1) + ((1-ppctr) *(1-p1)/(1-r1)))
      }
    }
    else{
      udf{prob: Vector =>
        val ppctr = prob(i)
        (ppctr *p1/r1) / ((ppctr *p1/r1) + ((1-ppctr) *(1-p1)/(1-r1)))
      }
    }
    dataset.withColumn($(outputCol), scale(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ProbabilityTransformModel = {
    val copied = new ProbabilityTransformModel(uid, actualProportionOfClicks, underSampledProportionOfClicks, probFieldIndex)
      .setParent(parent)
    copyValues(copied, extra)
  }
}
