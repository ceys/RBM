package com.jd.ajmd.dp

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import breeze.linalg.operators._
import breeze.stats.distributions._
import scala.math.{log => mathLog}
import scala.collection.mutable.HashMap
import com.jd.ajmd.util.MathUtil

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
  nRating: Int,
  learningRate: Double) extends Serializable with Logging {

  // Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  // sampled from gaussian distribution.
  // the first row and first column represent the weights of bias

  val weights = new Array[Array[Array[Double]]](nVisible)
  val vbias = new Array[Array[Double]](nVisible)
  val hbias = new Array[Double](nHidden)

  // TODO: use fastutil to replace map. http://fastutil.di.unimi.it/
  private def initWeightsAndBias(data: RDD[Map[(Int, Int), Int]]): Int = {
    // "A practical guide to training restricted boltzmann machines" 8
    val gau = new Gaussian(0.0, 0.01)
    for (i <- 0 until nVisible) {
      for (j <- 0 until nHidden) {
        weights(i)(j) = Array.fill(nRating)(gau.sample())
      }
      vbias(i) = Array.fill(nRating)(0)
    }
    // visible bias: log[p_i/(1-p_i)] where p_i is the proportion of training vector in which unit i is on.
    val pmap = data.flatMap(_.toSeq).countByKey()
    val total = pmap.size
    pmap.foreach( pair => vbias(pair._1._1)(pair._1._2) = mathLog(pair._2.toDouble/(total-pair._2)))

    // hidden bias: Not using a sparsity target.
    for (j <- 0 until nHidden) hbias(j) = 0.0

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
    var globalHbias = sc.broadcast(hbias)
    var globalVbias = sc.broadcast(vbias)
    val globalGradient = sc.broadcast(new Array[Array[Array[Double]]](nVisible))

    for (t <- 1 to numIterations) {
      val w = data.mapPartitions(it => {
        val gradient = new Array[Array[Array[Double]]](nVisible)
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

  private def computeGradient(input: Map[(Int, Int), Int],
                              hBias: Array[Double],
                              vBias: Array[Array[Double]],
                              weight: Array[Array[Array[Double]]],
                              gradient: Array[Array[Array[Double]]]) = {

    val posHiddenActivations = hBias.clone()
    for (j <- 0 until nHidden) {
      input.foreach {
        case(k, v) => posHiddenActivations(j) += v * weight(k._1)(j)(k._2)
      }
    }
    val posHiddenProbs = posHiddenActivations.map(sigmoid(_))
    val posHiddenStates = posHiddenProbs.map(MathUtil.binomial(_))

    val negVisibleActivations = input.map{ case(k, v) => (k._1, vBias(k._1)) }
    input.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          for (kr <- 0 until nRating) {
            negVisibleActivations(k._1)(kr) += weight(k._1)(j)(kr) * posHiddenStates(j)
          }
        }
      }
    }
    val negVisibleProbs = negVisibleActivations.map { case(k, v) => MathUtil.softMax(v) }
    val negVisibleStats = negVisibleProbs.map { case(k, v) => MathUtil.}



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
