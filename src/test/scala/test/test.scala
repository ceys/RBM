package test


import scala.math.random

import org.apache.spark.Logging
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object test extends java.io.Serializable with Logging{
  def main(args: Array[String]) {
    println("start")
    val numClass = args(0).toInt
    val numDim = args(1).toInt
    val learningRate = args(2).toDouble
    val maxIter = args(3).toInt
    val numSamples = args(4).toInt
    val partitions = args(5).toInt


    val conf = new SparkConf().setAppName("testSoftmax").setMaster("local")
    val spark = new SparkContext(conf)

    val samples = spark.parallelize(0 until numSamples, partitions).map{ case i =>
      val cls = i % numClass
      var feature = new Array[Double](numDim)
      new Sample(feature.map(_ => (cls + 1) * 10.0 + random), cls)
    }
    println(samples.count)

    val model = Softmax.train(samples, numClass, numDim, learningRate, maxIter)
    spark.parallelize(model.weights.zipWithIndex, partitions).map{case (ws, cls) =>
      var str = cls.toString
      for (i <- 0 until ws.length) str += " " + ws(i).toString
    }//.saveAsTextFile("/tmp/Softmax_weights")

    spark.stop()

    println("finish")
  }
}
