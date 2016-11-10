package org.apache.spark.ml.feature

import scala.collection.mutable.ArrayBuilder

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.BinaryAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * Created by karan.verma on 09/11/16.
 */
class VectorBinarizer(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("vectorBinarizer"))

  /**
   * Param for threshold used to binarize Vector of continuous features.
   * The features greater than the threshold, will be binarized to 1.0.
   * The features equal to or less than the threshold, will be binarized to 0.0.
   * Default Threshold: 0.0
   * @group param
   */
  val threshold: DoubleParam =
    new DoubleParam(this, "threshold", "threshold used to binarize continuous features contained in a vector")

  /** @group getParam */
  def getThreshold: Double = $(threshold)

  /** @group setParam */
  def setThreshold(value: Double): this.type = {
    if (value < 0.0)
      throw new IllegalArgumentException("Do not support negative value")
    else
      set(threshold, value)
  }

  setDefault(threshold -> 0.0)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = true)
    val schema = dataset.schema
    val inputType = schema($(inputCol)).dataType
    val td = $(threshold)

    val binarizerVector = udf { (data: Vector) =>

      val indices = ArrayBuilder.make[Int]
      val values = ArrayBuilder.make[Double]



      data.foreachActive { (index, value) =>
        if (value > td) {
          values +=  1.0
          indices += index
        }
        else {
          values += 0.0
          indices += index
        }
      }

      data match {
        case _:DenseVector =>
          Vectors.sparse(data.size, indices.result(), values.result()).toDense
        case _:SparseVector =>
          Vectors.sparse(data.size, indices.result(), values.result())
        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }

    }

    val metadata = outputSchema($(outputCol)).metadata

    inputType match {
      case _: VectorUDT =>
        dataset.select(col("*"), binarizerVector(col($(inputCol))).as($(outputCol), metadata))
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    val outputColName = $(outputCol)

    val outCol: StructField = inputType match {
      case _: VectorUDT =>
        new StructField(outputColName, new VectorUDT, true)
      case other =>
        throw new IllegalArgumentException(s"Data type $other is not supported.")
    }

    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists.")
    }
    StructType(schema.fields :+ outCol)
  }

  override def copy(extra: ParamMap): VectorBinarizer = defaultCopy(extra)
}

object VectorBinarizer extends DefaultParamsReadable[VectorBinarizer] {

  override def load(path: String): VectorBinarizer = super.load(path)
}

