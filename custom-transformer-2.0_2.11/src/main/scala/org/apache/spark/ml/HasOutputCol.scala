package org.apache.spark.ml

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by vinay.varma on 08/11/16.
  */
trait HasOutputCol extends Params {
  val outputCol: Param[String] = new Param[String](this, "outputCol", "Will have the fraction of common words.")

  def getOutputCol = $(outputCol)
}
