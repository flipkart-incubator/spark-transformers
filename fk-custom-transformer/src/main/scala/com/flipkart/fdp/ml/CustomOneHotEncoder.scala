package com.flipkart.fdp.ml

/**
 * Created by shubhranshu.shekhar on 21/06/16.
 */
import org.apache.spark.annotation.Experimental
import org.apache.spark.annotation.Since
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.Model
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.attribute.BinaryAttribute
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.attribute._
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._

import scala.collection.mutable

/**
 * Created by shubhranshu.shekhar on 20/06/16.
 */

trait CustomOneHotParams extends Params {
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)
  setDefault(inputCol, uid + "__input")

  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  setDefault(outputCol, uid + "__output")
  final def getOutputCol: String = $(outputCol)

  //val vectorSize: IntParam =
  //  new IntParam(this, "Vector Size", "size of the vector", ParamValidators.gt(0))

  /** @group getParam */
  //def getVectorSize: Int = $(vectorSize)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    //CustomOneHotSchemaUtil.checkColumnTypes(schema, $(inputCol), typeCandidates)
    CustomSchemaUtil.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}


@Experimental
class CustomOneHotEncoder(override val uid: String)
  extends Estimator[CustomOneHotEncoderModel] with CustomOneHotParams
{

  def this() = this(Identifiable.randomUID("customOneHot"))

  /**
   * Whether to drop the last category in the encoded vector (default: true)
   * @group param
   */
  final val dropLast: BooleanParam =
    new BooleanParam(this, "dropLast", "whether to drop the last category")
  setDefault(dropLast -> true)

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)


  /** @group setParam */
  def setDropLast(value: Boolean): this.type = set(dropLast, value)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    require(schema(inputColName).dataType.isInstanceOf[NumericType],
      s"Input column must be of type NumericType but got ${schema(inputColName).dataType}")
    val inputFields = schema.fields
    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    val inputAttr = Attribute.fromStructField(schema(inputColName))
    val outputAttrNames: Option[Array[String]] = inputAttr match {
      case nominal: NominalAttribute =>
        if (nominal.values.isDefined) {
          nominal.values
        } else if (nominal.numValues.isDefined) {
          nominal.numValues.map(n => Array.tabulate(n)(_.toString))
        } else {
          None
        }
      case binary: BinaryAttribute =>
        if (binary.values.isDefined) {
          binary.values
        } else {
          Some(Array.tabulate(2)(_.toString))
        }
      case _: NumericAttribute =>
        throw new RuntimeException(
          s"The input column $inputColName cannot be numeric.")
      case _ =>
        None // optimistic about unknown attributes
    }

    val filteredOutputAttrNames = outputAttrNames.map { names =>
      if ($(dropLast)) {
        require(names.length > 1,
          s"The input column $inputColName should have at least two distinct values.")
        names.dropRight(1)
      } else {
        names
      }
    }

    val outputAttrGroup = if (filteredOutputAttrNames.isDefined) {
      val attrs: Array[Attribute] = filteredOutputAttrNames.get.map { name =>
        BinaryAttribute.defaultAttr.withName(name)
      }
      new AttributeGroup($(outputCol), attrs)
    } else {
      new AttributeGroup($(outputCol))
    }

    val outputFields = inputFields :+ outputAttrGroup.toStructField()
    StructType(outputFields)
  }


  override def fit(dataFrame: DataFrame): CustomOneHotEncoderModel = {//Dataset[_]
  //val dataFrame = dataset.toDF()
  // schema transformation
  val inputColName: String = $(inputCol)
    val outputColName: String = $(outputCol)
    val shouldDropLast = $(dropLast)
    var outputAttrGroup = AttributeGroup.fromStructField(
      transformSchema(dataFrame.schema)(outputColName))
    if (outputAttrGroup.size < 0) {
      // If the number of attributes is unknown, we check the values from the input column.
      val numAttrs = dataFrame.select( col(inputColName).cast(DoubleType) ).rdd.map(_.getDouble(0))
        .aggregate(0.0)(
          (m, x) => {
            assert(x <= Int.MaxValue,
              s"OneHotEncoder only supports up to ${Int.MaxValue} indices, but got $x")
            assert(x >= 0.0 && x == x.toInt,
              s"Values from column $inputColName must be indices, but got $x.")
            math.max(m, x)
          },
          (m0, m1) => {
            math.max(m0, m1)
          }
        ).toInt + 1
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val filtered = if (shouldDropLast) outputAttrNames.dropRight(1) else outputAttrNames
      val outputAttrs: Array[Attribute] =
        filtered.map(name => BinaryAttribute.defaultAttr.withName(name))
      outputAttrGroup = new AttributeGroup(outputColName, outputAttrs)
    }
    // data transformation
    val size = outputAttrGroup.size
    require(size > 0, "The vector size should be > 0")

    return new CustomOneHotEncoderModel(uid, size).setInputCol(inputColName).setOutputCol(outputColName)
    //dataFrame.select(col("*"), encode(col(inputColName).cast(DoubleType)).as(outputColName, metadata))
  }

  override def copy(extra: ParamMap): CustomOneHotEncoder = defaultCopy(extra)
}


class CustomOneHotEncoderModel(override val uid: String, val vectorSize: Int)
  extends Model[CustomOneHotEncoderModel] with CustomOneHotParams {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataFrame: DataFrame): DataFrame = {
    transformSchema(dataFrame.schema, logging = true)
    val oneValue = Array(1.0)
    val emptyValues = Array[Double]()
    val emptyIndices = Array[Int]()
    val encode = udf { label: Double =>
      if (label < vectorSize) {
        Vectors.sparse(vectorSize, Array(label.toInt), oneValue)
      } else {
        Vectors.sparse(vectorSize, emptyIndices, emptyValues)
      }
    }
    dataFrame.withColumn($(outputCol), encode(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): CustomOneHotEncoderModel = {
    val copied = new CustomOneHotEncoderModel(uid, vectorSize).setParent(parent)
    copyValues(copied, extra)
  }
}


