package com.flipkart.fdp.ml

import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.param.{ParamMap, Param, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vectors, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

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

  /** Validates and transforms the input schema. */
  protected def transformSchema(schema: StructType): StructType = {
    //val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    //CustomSchemaUtil.checkColumnTypes(schema, $(inputCol), typeCandidates)
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    CustomSchemaUtil.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

class ProbabilityTransform(override val uid: String) extends Estimator[ProbabilityTransformModel]
  with ProbabilityTransformParams{
  def this() = this(Identifiable.randomUID("probabilityTransform"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setActualClickProportion(value: Double): this.type = set(actualClickProportion, value)

  def setUnderSampledClickProportion(value: Double): this.type = set(underSampledClickProportion, value)

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def fit(dataset: DataFrame): ProbabilityTransformModel = {
    transformSchema(dataset.schema)
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    val p1 = $(actualClickProportion)
    val r1 = $(underSampledClickProportion)

    copyValues(new ProbabilityTransformModel(uid, p1, r1).setInputCol(inputColName).setOutputCol(outputColName).setParent(this))
  }

  override def copy(extra: ParamMap): AlgebraicTransform = defaultCopy(extra)

}

class ProbabilityTransformModel(override val uid: String, val actualProportionOfClicks: Double,
                                val underSampledProportionOfClicks: Double)
  extends Model[ProbabilityTransformModel] with ProbabilityTransformParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val p1 = actualProportionOfClicks
    val r1 = underSampledProportionOfClicks
    val scale = udf{ ppctr: Double =>
      (ppctr *p1/r1) / ((ppctr *p1/r1) + ((1-ppctr) *(1-p1)/(1-r1)))
    }
    dataset.withColumn($(outputCol), scale(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    transformSchema(schema)
  }

  override def copy(extra: ParamMap): ProbabilityTransformModel = {
    val copied = new ProbabilityTransformModel(uid, actualProportionOfClicks, underSampledProportionOfClicks)
      .setParent(parent)
    copyValues(copied, extra)
  }
}
