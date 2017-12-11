package org.apache.spark.ml

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._

class StringSanitizer(override val uid: String) extends UnaryTransformer[String, Seq[String], StringSanitizer] with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("StringSanitizer"))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, true)

  override def copy(extra: ParamMap): StringSanitizer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields :+ StructField(getOutputCol, StringType))
  }

  override protected def createTransformFunc: (String) => Seq[String] = { originStr =>
    originStr.toLowerCase
      .replaceAll("\\P{Print}", " ")
      .replaceAll("[^0-9a-zA-Z ]", " ")
      .replaceAll("\\d{10}", " ")
      .replaceAll("\\d{6}", " ")
      .trim
      .replaceAll("""\s+""", " ")
      .split(" ")
  }
}

object StringSanitizer extends DefaultParamsReadable[StringSanitizer]