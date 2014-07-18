package com.jd.ajmd.dp

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import breeze.linalg.operators._
import breeze.stats.distributions._
import scala.math.{log => mathLog}

/**
 * Created by zhengchen on 14-7-10.
 */

class RbmModel (
  val weights: Array[Array[Array[Double]]],
  val vbias: Array[Array[Double]],
  val hbias: Array[Array[Double]]) extends Serializable


class RbmWithCD (
  nVisible: Int,
  nHidden: Int,
  k: Int,
  learningRate: Double) extends Serializable with Logging {

  // Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  // sampled from gaussian distribution.
  // the first row and first column represent the weights of bias

  val weights = new Array[Array[Array[Double]]](nVisible)
  val vbias = new Array[Array[Double]](nVisible)
  val hbias = new Array[Array[Double]](nHidden)

  // TODO: use fastutil to replace map. http://fastutil.di.unimi.it/
  private def initWeightsAndBias(data: RDD[Map[(Int, Int), Int]]): Int = {
    // "A practical guide to training restricted boltzmann machines" 8
    val gau = new Gaussian(0.0, 0.01)
    for (i <- 0 until nVisible) {
      for (j <- 0 until nHidden) {
        weights(i)(j) = Array.fill(k)(gau.sample())
      }
      vbias(i) = Array.fill(k)(0)
    }
    // visible bias: log[p_i/(1-p_i)] where p_i is the proportion of training vector in which unit i is on.
    val pmap = data.flatMap(_.toSeq).countByKey()
    val total = pmap.size
    pmap.foreach( pair => vbias(pair._1._1)(pair._1._2) = mathLog(pair._2.toDouble/(total-pair._2)))

    // hidden bias: Not using a sparsity target.
    for (j <- 0 until nHidden) hbias(j) = Array.fill(k)(0)

    total
  }

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
  def run(sc: SparkContext, data: RDD[Map[(Int, Int), Int]], numIterations: Int, p: Int): RbmModel = {
    val numData = initWeightsAndBias(data)
    var globalWeights = sc.broadcast(weights)
    var globalGradient = sc.accumulable()

    for (t <- 1 to numIterations) {
      val w = data.mapPartitions(it => {
        val gradient = computeGradient(it.next(), globalWeights.value)
        while (it.hasNext) {
          gradient += computeGradient(it.next(), globalWeights.value)
        }
        Iterator.single(gradient)
      }).reduce(_+_)

      weights += (w.toDenseMatrix / numData) * learningRate
      globalWeights = sc.broadcast(weights)

    }
    new RbmModel(weights)
  }


  private def computeGradient(input: Map[(Int, Int), Int], weight: DenseMatrix[Double]): DenseMatrix[Double] = {

    // Clamp to the data and sample from the hidden units.
    // breeze dose not support the operation of vector-matrix production. WTF!
    // val posHiddenActivations = dotProduct(input, weight)
    val inputIndex = input.index

    val posHiddenActivations = (weight.t * input).toDenseVector
    val posHiddenProbs = sigmoid(posHiddenActivations)
    val posHiddenStats = I((posHiddenProbs :> DenseVector.rand(nHidden+1)).toDenseVector)

    // Reconstruct the visible units and sample again from the hidden units.

    val builder = new VectorBuilder[Double](weight.rows)
    for (i <- 0 until inputIndex.length) builder.add(inputIndex(i), weight(inputIndex(i), ::) * posHiddenStats)
    val negVisibleActivations = builder.toSparseVector
    //val negVisibleActivations = (posHiddenStats.asDenseMatrix * weight(inputIndex, ::).t).toDenseVector
    println(negVisibleActivations)

    builder.clear()
    val it = negVisibleActivations.activeIterator
    while (it.hasNext) {
      val a = it.next()
      builder.add(a._1, sigmoid(a._2))
    }
    val negVisibleProbs = builder.toSparseVector()
    //val negVisibleProbs = sigmoid(negVisibleActivations)
    println(negVisibleProbs)
    // Fix the bias unit.
    negVisibleProbs(0) = 1.0

    val negHiddenActivations = DenseVector.zeros[Double](weights.cols)
    for (i <- 0 until inputIndex.length) negHiddenActivations += weight(inputIndex(i), ::).t * negVisibleProbs(i)
    //val negHiddenActivations = (negVisibleProbs * weight).toDenseVector
    val negHiddenProbs = sigmoid(negHiddenActivations)

    // the sparseVector col vector cannot dot denseVector row vector. WTF!
    //input.toDenseVector * posHiddenProbs.t - (negVisibleProbs * negHiddenProbs.t)
    println(posHiddenProbs)
    println(input)
    println(negHiddenProbs)
    println(negVisibleProbs)
    println(posHiddenProbs.asDenseMatrix.t * input.t)
    println(negHiddenProbs.asDenseMatrix.t * negVisibleProbs.t)
    (posHiddenProbs.asDenseMatrix.t * input.t - negHiddenProbs.asDenseMatrix.t * negVisibleProbs.t).t
    //input * posHiddenProbs
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
                     new SparseVector[Double](Array(0,3,4,5), Array(1.0,1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,3,4), Array(1.0,1.0,1.0), 7),
                     new SparseVector[Double](Array(0,3,4,5), Array(1.0,1.0,1.0,1.0), 7))

    val numParallel = 1

    val conf = new SparkConf().setAppName("rbm").setMaster("local")
    val sc = new SparkContext(conf)
    val input = sc.parallelize(test, numParallel)
    val m = train(sc, input, 100, 6, 2, 0.1, 1)
    //val m = train(sc, input, 100, 784, 500, 0.1, 1, numParallel)
    println(m.weights)
    //println {toArray.map(t => t.mkString(" ")).mkString("\n") }
  }

}
