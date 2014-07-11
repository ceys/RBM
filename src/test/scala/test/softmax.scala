package test

import scala.math.exp

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.Logging


class SoftmaxModel(
                    var weights: Array[Array[Double]]) extends java.io.Serializable with Logging{
  def predict(feature: Array[Double]): Int={
    0
  }
}

case class Sample(feature: Array[Double], cls: Int)

object Softmax extends Serializable with Logging{

  private def gradientInc(
                           sample: Sample,
                           gradientMatrix: Array[Array[Double]],
                           ws: Array[Array[Double]],
                           numClass: Int,
                           numDim: Int) =
  {
    val feature = sample.feature
    val cls = sample.cls

    var p = new Array[Double](numClass)
    for (i <- 0 until numClass){
      for (j <- 0 until numDim){
        p(i) += ws(i)(j) * feature(j)
      }
      p(i) += ws(i)(numDim)
    }
    p = p.map(exp(_))
    val sum = p.sum
    p = p.map(_ / sum)
    p(cls) -= 1.0

    for (i <- 0 until numClass){
      for (j <- 0 until numDim){
        gradientMatrix(i)(j) += feature(j) * p(i)
      }
      gradientMatrix(i)(numDim) += p(i)
    }
  }

  private def plus(matrixA: Array[Array[Double]], matrixB: Array[Array[Double]])
  : Array[Array[Double]] = {
    for (i <- 0 until matrixA.length)
      for(j <- 0 until matrixA.head.length)
        matrixA(i)(j) += matrixB(i)(j)
    matrixA
  }

  private def parameterUpdate(
                               weights: Array[Array[Double]],
                               gradient: Array[Array[Double]],
                               numSample: Long) =
  {
    for (i <- 0 until weights.length)
      for(j <- 0 until weights.head.length)
        weights(i)(j) -= gradient(i)(j) / numSample
  }

  /*  val func = (it: Iterator[Sample]) => {
          logInfo("111111111111")
          var gradientMatrix = new Array[Array[Double]](this.numClass)
          for (i <- 0 until numClass) gradientMatrix(i) = new Array[Double](numDim + 1)
          it.map{sample => gradientInc(sample, gradientMatrix, globalWeights.value)}
          println(gradientMatrix.length)
          println(gradientMatrix.head.length)
          Iterator.single(gradientMatrix)
          }
  */

  def run(samples: RDD[Sample], numClass: Int, numDim: Int, maxIter: Int, learningRate: Double) =
  {
    logInfo("111111111111")
    val sc = samples.context
    //weights broadcast
    var weights = new Array[Array[Double]](numClass)
    for (i <- 0 until numClass) weights(i) = new Array[Double](numDim + 1)
    logInfo("111111111111")


    logInfo("111111111111")
    var globalNumClass = sc.broadcast(numClass)
    var globalNumDim = sc.broadcast(numDim)

    samples.cache()
    for (iter <- 0 until maxIter)
    {
      println(samples.count)
      val cls = numClass
      val dim = numDim
      val ww = weights
    var globalWeights = sc.broadcast(weights)
      //gradient foreach partition
      val gradient = samples.mapPartitions({it =>
        var gradientMatrix = new Array[Array[Double]](cls)
        //for (i <- 0 until numClass) gradientMatrix(i) = new Array[Double](dim + 1)
        val ws = globalWeights.value
        //it.map{sample => gradientInc(sample, gradientMatrix, globalWeights.value)}
        //logInfo("111111111111")
        //println(gradientMatrix.length)
        //println(gradientMatrix.head.length)
        //Iterator.single(gradientMatrix)
        //Iterator.single(3)
        it
      }, true).collect//reduce{(matrixA, matrixB) => matrixA}//plus(matrixA, matrixB)}
      //update
      //parameterUpdate(weights, gradient, samples.count)

      globalWeights = sc.broadcast(weights)
    }
    new SoftmaxModel(weights)
  }

  def train(
             samples: RDD[Sample],
             numClass: Int,
             numDim: Int,
             learningRate: Double,
             maxIter: Int
             ): SoftmaxModel = {
    run(samples, numClass, numDim, maxIter, learningRate)
  }
}
