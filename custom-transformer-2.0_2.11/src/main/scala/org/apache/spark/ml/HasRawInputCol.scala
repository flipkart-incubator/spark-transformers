package org.apache.spark.ml

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by vinay.varma on 08/11/16.
  */
trait HasRawInputCol extends Params {
  val rawInputCol: Param[String] = new Param[String](this, "rawInputCol", "Raw words.")

  def getRawInputCol = $(rawInputCol)
}
