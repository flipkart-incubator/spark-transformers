package com.flipkart.transformer.common

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by gaurav.prasad on 08/11/16.
  */
/*
* Transformers and Estimators
* */
trait HasInputCol extends Params {
  val inputCol: Param[String] = new Param[String](this, "inputCol", "Input should have sanitized split words.")

  def getInputCol = $(inputCol)
}
