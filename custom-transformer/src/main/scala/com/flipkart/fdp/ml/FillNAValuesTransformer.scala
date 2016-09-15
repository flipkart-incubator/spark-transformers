package com.flipkart.fdp.ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType


class FillNAValuesTransformer(override val uid: String) extends Transformer {

  val naValueMap: Param[java.util.Map[String, Any]] = new Param[java.util.Map[String, Any]](this, "naValueMap", "column name to default value map in case value is NA");
  setDefault(naValueMap -> new java.util.HashMap[String, Any]())

  def this() {
    this(Identifiable.randomUID("FillNAValuesTransformer"))
  }

  def getNAValueMap: java.util.Map[String, Any] = $(naValueMap)

  def setNAValueMap(columnToNAValueMap: java.util.Map[String, Any]) = {
    if(! columnToNAValueMap.isEmpty) {
      $(naValueMap).putAll(columnToNAValueMap);
    }
  }

  override def transform(dataFrame: DataFrame): DataFrame = {
    dataFrame.na.fill($(naValueMap));
  }

  override def copy(extra: ParamMap): FillNAValuesTransformer = {
    val copied = new FillNAValuesTransformer(uid)
    copyValues(copied, extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    //This Transformer does not change the schema of df
    return schema;
  }

}
