package org.apache.spark.ml

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * Created by vinay.varma on 08/11/16.
  */
class CommonAddressFeatures(override val uid: String) extends Transformer with HasInputCol with HasRawInputCol with DefaultParamsWritable {

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setRawInputCol(value: String): this.type = set(rawInputCol, value)

  setDefault(inputCol, "sanitizedAddress")
  setDefault(rawInputCol, "mergedAddress")

  val numWordsParam: Param[String] = new Param[String](this, "numWords", "No. of words in cleaned address.")
  def getNumWordsParam = $(numWordsParam)

  val numCommasParam: Param[String] = new Param[String](this, "numCommas", "No. of commas in raw address.")
  def getNumCommasParams = $(numCommasParam)

  val numericPresentParam: Param[String] = new Param[String](this, "numericPresent", "Numbers present in cleaned address.")
  def getNumericPresentParam = $(numericPresentParam)

  val addressLengthParam: Param[String] = new Param[String](this, "addressLength", "Length of cleaned address.")
  def getAddressLengthParam = $(addressLengthParam)

  val favouredStartColParam: Param[String] = new Param[String](this, "favouredStartColParam", "name of col having fav start res.")
  def getFavouredStartColParam = $(favouredStartColParam)

  val unfavouredStartColParam: Param[String] = new Param[String](this, "unfavouredStartColParam", "name of col having un-fav start res.")
  def getUnfavouredStartColParam = $(unfavouredStartColParam)

  val favourableStartsParam = new StringArrayParam(this, "favourableStarts", "Addresses starting with these are good.")
  val unFavourableStartsParam = new StringArrayParam(this, "unFavourableStarts", "Addresses starting with these are bad.")

  def this() = this(Identifiable.randomUID("CommonAddressFeatures"))

  val favourableStartWords = Array("plot", "flat", "house", "room")
  val unfavourableStartWords =Array("near", "opp", "opposite")

  setDefault(numWordsParam, "numWords")
  setDefault(numCommasParam, "numCommas")
  setDefault(numericPresentParam, "numericPresent")
  setDefault(addressLengthParam, "addressLength")
  setDefault(favourableStartsParam, favourableStartWords)
  setDefault(unFavourableStartsParam, unfavourableStartWords )
  setDefault(favouredStartColParam, "favouredStart")
  setDefault(unfavouredStartColParam, "unfavouredStart")

  def getOutputParams:Seq[Param[String]] = Seq(numWordsParam, numCommasParam, numericPresentParam, addressLengthParam, favouredStartColParam, unfavouredStartColParam)

  def wordCount = udf((words: Seq[String]) => {
    words.length
  })

  def commaCount = udf((rawLine: String) => rawLine.split(",").length - 1)

  def numericPresent = udf { (words: Seq[String]) =>
    "[0-9]".r.findFirstIn(words.mkString(" ")) match {
      case Some(_) => 1.0
      case None => 0.0
    }
  }

  def favouredStart = udf { (words: Seq[String]) =>
    $(favourableStartsParam).contains(words.head) match {
      case true =>
        1.0
      case false =>
        0.0
    }
  }

  def unfavouredStart = udf { (words: Seq[String]) =>
    $(unFavourableStartsParam).contains(words.head) match {
      case true =>
        1.0
      case false =>
        0.0
    }
  }

  def addressLength = udf((words: Seq[String]) => words.mkString(" ").length)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset
      .withColumn($(numWordsParam), wordCount(col($(inputCol))).cast(DoubleType))
      .withColumn($(numCommasParam), commaCount(col($(rawInputCol))).cast(DoubleType))
      .withColumn($(numericPresentParam), numericPresent(col($(inputCol))))
      .withColumn($(addressLengthParam), addressLength(col($(inputCol))).cast(DoubleType))
      .withColumn($(favouredStartColParam), favouredStart(col($(inputCol))))
      .withColumn($(unfavouredStartColParam), unfavouredStart(col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    StructType(
      schema.fields
        :+ StructField($(numWordsParam), DoubleType)
        :+ StructField($(numCommasParam), DoubleType)
        :+ StructField($(numericPresentParam), DoubleType)
        :+ StructField($(addressLengthParam), DoubleType)
        :+ StructField($(favouredStartColParam), DoubleType)
        :+ StructField($(unfavouredStartColParam), DoubleType)
    )
  }
}

object CommonAddressFeatures extends DefaultParamsReadable[CommonAddressFeatures]