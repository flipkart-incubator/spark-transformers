package com.flipkart.fdp.ml

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Transformer}
import org.apache.spark.ml.param.{ParamMap, Param, Params}
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructType}
import org.apache.spark.sql.functions.{col, udf, lit}


/**
 * Created by shubhranshu.shekhar on 13/07/16.
 */

trait AlgebraicTransformParams extends Params {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  setDefault(outputCol, uid + "__output")
  final def getOutputCol: String = $(outputCol)

  final val coefficients: Param[Array[Double]] = new Param[Array[Double]](this, "coefficients", "algebraic coefficient")
  final def getCoefficients: Array[Double] = $(coefficients)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    //val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    //CustomSchemaUtil.checkColumnTypes(schema, $(inputCol), typeCandidates)
    CustomSchemaUtil.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
 * Algebraically transforms a scaler. Coeffiecients are mentioned in increasing order of degrees.
 * e.g x = 3.5, Ax + b => coefficients = [b, A], value = x
 */
class AlgebraicTransform(override val uid: String) extends Transformer with AlgebraicTransformParams{
  def this() = this(Identifiable.randomUID("algebraicTransform"))
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setCoefficients(value: Array[Double]): this.type = set(coefficients, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema)
    val coeff = $(coefficients)
    val encode = udf {(value: Double) =>
      if(coeff.length == 0){
        0.0
      }
      else{
        var sum = coeff(0)
        var mul = value
        for(i <- 1 until coeff.length){
          sum = sum + coeff(i) * mul
          mul = mul * value;
        }
        sum
      }
    }
    //In our version of spark Transformer treats DataFrame === Dataset
    dataset.withColumn($(outputCol), encode(col($(inputCol))))
  }

  override def copy(extra: ParamMap): AlgebraicTransform = defaultCopy(extra)

  /*override def copy(extra: ParamMap): ProbabilityTransformModel = {
    val copied = new ProbabilityTransformModel(uid, actualProportionOfClicks, underSampledProportionOfClicks, probIndex)
      .setParent(parent)
    copyValues(copied, extra)
  }*/
}
