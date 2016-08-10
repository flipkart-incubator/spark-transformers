package com.flipkart.fdp.ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType


class CustomLogScaler(override val uid: String, val addValue: Double)
  extends Transformer {

  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def this(addValue: Double) {
    this(Identifiable.randomUID("customLogScaler"), addValue)
  }

  final def getInputCol: String = $(inputCol)

  def setInputCol(value: String): this.type = set(inputCol, value)

  final def getOutputCol: String = $(outputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataFrame: DataFrame): DataFrame = {
    transformSchema(dataFrame.schema)

    val encode = udf { inputVector: Vector =>
      inputVector match {
        case DenseVector(vs) =>
          val values = vs.clone()
          val size = values.size
          var i = 0
          while (i < size) {
            values(i) = Math.log(addValue + values(i));
            i += 1
          }
          Vectors.dense(values)
        case SparseVector(size, indices, vs) =>
          // For sparse vector, the `index` array inside sparse vector object will not be changed,
          // so we can re-use it to save memory.
          val values = vs.clone()
          val nnz = values.size
          var i = 0
          while (i < nnz) {
            values(i) = Math.log(addValue + values(i));
            i += 1
          }
          Vectors.sparse(size, indices, values)
        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }
    }
    dataFrame.withColumn($(outputCol), encode(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    require(inputType.isInstanceOf[VectorUDT],
      s"Input column ${$(inputCol)} must be a vector column")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    return CustomSchemaUtil.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  override def copy(extra: ParamMap): CustomLogScaler = {
    val copied = new CustomLogScaler(uid, addValue)
    copyValues(copied, extra)
  }
}

