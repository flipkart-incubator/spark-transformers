package org.apache.spark.ml

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWriter, _}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * Created by vinay.varma on 08/11/16.
  */
class PopularWordsEstimator(override val uid: String) extends Estimator[PopularWordsModel] with HasInputCol
  with HasOutputCol {
  val maxPopularWords: Param[Int] = new Param[Int](this, "maxPopularWords", "n popular words.")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol, "sanitizedAddress")
  setDefault(outputCol, "commonFraction")

  def this() = this(Identifiable.randomUID("PopularWordsEstimator"))
  setDefault(maxPopularWords, 200)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    //    check input col is of type string
    StructType(schema.fields :+ StructField($(outputCol), DoubleType))
  }

  override def fit(dataset: Dataset[_]): PopularWordsModel = {
    val commonWords = dataset
      .where(col("label") === 0.0)
      .select($(inputCol))
      .rdd.flatMap(r => r.getAs[Seq[String]]($(inputCol)))
      .map((_, 1))
      .reduceByKey(_ + _)
      .takeOrdered($(maxPopularWords))(Ordering[Int].reverse.on(_._2))
      .map(_._1)
    copyValues(new PopularWordsModel(uid, commonWords).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[PopularWordsModel] = defaultCopy(extra)
}

class PopularWordsModel(override val uid: String, val popularWords: Array[String]) extends Model[PopularWordsModel]
  with HasInputCol with HasOutputCol with MLWritable {

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def copy(extra: ParamMap): PopularWordsModel = defaultCopy(extra)

  def commonFraction = udf { (words: Seq[String]) =>
    val commonWords = words.filter(popularWords.contains(_))
    if( words.nonEmpty) {
      commonWords.length.toDouble / words.length
    }else {
      0.0
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn($(outputCol), commonFraction(col($(inputCol))))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    //    check type of input col and add col for output col.
    StructType(schema.fields :+ StructField($(outputCol), DoubleType))
  }

  override def write: MLWriter = PopularWordsModel.writer(this)
}

object PopularWordsModel extends MLReadable[PopularWordsModel] {
  private val className = classOf[PopularWordsModel].getName

  def read: MLReader[PopularWordsModel] = new PopularWordsModelReader


  def writer(model: PopularWordsModel) = new PopularWordsModelWriter(model)

  class PopularWordsModelWriter(instance: PopularWordsModel) extends MLWriter {
    def saveImpl(path: String): Unit = {
      /*
      * Whole point of using this package is to use this method.
      * */
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.popularWords)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  class PopularWordsModelReader extends MLReader[PopularWordsModel] {
    override def load(path: String): PopularWordsModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(popularWords: Seq[String]) = data.select("popularWords").head()
      val model = new PopularWordsModel(metadata.uid, popularWords.toArray)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  private case class Data(popularWords: Array[String])


}

