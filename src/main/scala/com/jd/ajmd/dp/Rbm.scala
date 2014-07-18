package com.jd.ajmd.dp

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import breeze.linalg.operators._

/**
 * Created by zhengchen on 14-7-10.
 */

class RbmModel (
  val weights: DenseMatrix[Double]) extends Serializable


class RbmWithCD (
  nVisible: Int,
  nHidden: Int,
  learningRate: Double) extends Serializable with Logging {

  // Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  // sampled from uniform distribution.
  // the first row and first column represent the weights of bias
  val weights = DenseMatrix.rand(nVisible + 1, nHidden + 1)

  /**
   * Run RBM with parameters on an 2-dim Array containing input data.
   *
   * @param sc - spark context
   *
   *
   * @param data - input data
   * @param numIterations - max num of iterations
   * @param p - num of partitions
   *
   * @return a RbmModel with edge-weights and bias.
   */
  def run(sc: SparkContext, data: RDD[SparseVector[Double]], numIterations: Int, p: Int): RbmModel = {
    var globalWeights = sc.broadcast(weights)
    val numData = data.count().toDouble
    var t = 0
    while (t < numIterations) {
      val w = data.mapPartitions(it => {
        val gradient = computGradient(it.next(), globalWeights.value)
        while (it.hasNext) {
          gradient += computGradient(it.next(), globalWeights.value)
        }
        Iterator.single(gradient)
      }).reduce(_+_)

      weights += (w / numData) * learningRate
      globalWeights = sc.broadcast(weights)

      t += 1
    }
    new RbmModel(weights)
  }


  private def computGradient(input: SparseVector[Double], weight: DenseMatrix[Double]): DenseMatrix[Double] = {

    // Clamp to the data and sample from the hidden units.
    // breeze dose not support the operation of vector-matrix production. WTF!
    // val posHiddenActivations = dotProduct(input, weight)
    val posHiddenActivations = (weight.t * input).toDenseVector
    val posHiddenProbs = sigmoid(posHiddenActivations)
    val posHiddenStats = I((posHiddenProbs :> DenseVector.rand(nHidden+1)).toDenseVector)

    // Reconstruct the visible units and sample again from the hidden units.
    val negVisibleActivations = (posHiddenStats.asDenseMatrix * weight.t).toDenseVector
    val negVisibleProbs = sigmoid(negVisibleActivations)
    // Fix the bias unit.
    negVisibleProbs(0) = 1.0

    val negHiddenActivations = (negVisibleProbs.asDenseMatrix * weight).toDenseVector
    val negHiddenProbs = sigmoid(negHiddenActivations)

    // the sparseVector col vector cannot dot denseVector row vector. WTF!
    //input.toDenseVector * posHiddenProbs.t - (negVisibleProbs * negHiddenProbs.t)
    (posHiddenProbs.asDenseMatrix.t * input.t).t - (negVisibleProbs * negHiddenProbs.t)
  }


  private def dotProduct(vector: Vector[Double], matrix: DenseMatrix[Double]): DenseVector[Double] = {
    val numHidden = matrix.cols
    val result = new Array[Double](numHidden)
    var a = 0
    while (a < numHidden) {
      result(a) = vector.dot(matrix(::, a))
      a += 1
    }
    new DenseVector[Double](result)
  }

}


object Rbm {

  def train(
             sc: SparkContext,
             input: RDD[SparseVector[Double]],
             numIterations: Int,
             nVisible: Int,
             nHidden: Int,
             learningRate: Double,
             p: Int): RbmModel = {
    new RbmWithCD(nVisible, nHidden, learningRate)
      .run(sc, input, numIterations, p)
  }

  def main(args: Array[String]) {
    //val test = Array(Array(1,1,1,0,0,0), Array(1,0,1,0,0,0), Array(1,1,1,0,0,0), Array(0,0,1,1,1,0), Array(0,0,1,1,0,0), Array(0,0,1,1,1,0))

    //import scala.io.Source
    //val test = Source.fromFile("test.txt").getLines().foreach(l => l.split(" ").foreach(s => s.toInt))
    val test = Array(new SparseVector[Double](Array(0,1,2,3), Array(1.0,1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,1,3), Array(1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,1,2,3), Array(1.0,1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,2,3,4), Array(1.0,1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,2,3), Array(1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,2,3,4), Array(1.0,1.0,1.0,1.0), 7))

    val numParallel = 1

    val conf = new SparkConf().setAppName("rbm").setMaster("local")
    val sc = new SparkContext(conf)
    val input = sc.parallelize(test, numParallel)
    val m = train(sc, input, 1000, 6, 2, 0.1, 1)
    //val m = train(sc, input, 100, 784, 500, 0.1, 1, numParallel)
    println(m.weights)
    //println {toArray.map(t => t.mkString(" ")).mkString("\n") }
  }

}
