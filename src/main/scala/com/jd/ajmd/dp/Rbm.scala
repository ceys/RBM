package com.jd.ajmd.dp

import org.apache.spark.{SparkContext, SparkConf, Logging}
import scala.util.Random
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext._

/**
 * Created by zhengchen on 14-7-10.
 */

class RbmModel (
  val weights: Array[Array[Double]],
  val vbias: Array[Double],
  val hbias: Array[Double]) extends Serializable


class RbmWithCD (
  nVisible: Int,
  nHidden: Int,
  learningRate: Double) extends Serializable with Logging{

  // Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  // sampled from uniform distribution.
  val weights = new Array[Array[Double]](nVisible)
  for (i <- 0 to nVisible-1) weights(i) = Array.fill(nHidden)(0.1 * Random.nextDouble())
  // Initialize hidden bias and visible bias sampled from uniform distribution.
  val hbias = Array.fill(nHidden)(0.1 * Random.nextDouble())
  val vbias = Array.fill(nVisible)(0.1 * Random.nextDouble())

  /**
   * Run RBM with parameters on an 2-dim Array containing input data.
   *
   * @param sc - spark context
   * @param data - input data
   * @param numIterations - max num of iterations
   * @param p - num of partitions
   * @param miniBatch - num of data should be used for one cd update
   *
   * @return a RbmModel with edge-weights and bias.
   */
  def run(sc: SparkContext, data: Array[Array[Int]], numIterations: Int, p: Int, miniBatch: Int): RbmModel = {
    // TODO: split data and broadcast
    val globalData = sc.broadcast(data)
    val dataLength = data.length

    // TODO: miniBatch
    for (t <- 1 to numIterations) {

      var d = 0
      while (d < dataLength) {

        // make new rdds with the current weights and bias.
        val weightsWithHiddenIndex = sc.parallelize(weights.transpose.zipWithIndex.map { case (a, i) => (i, a) }, p)
        val weightsWithVisibleIndex = sc.parallelize(weights.zipWithIndex.map { case (a, i) => (i, a) }, p)
        val hbiasWithIndex = sc.parallelize(hbias.zipWithIndex.map {case (h, i) => (i, h)}, p)
        val vbiasWithIndex = sc.parallelize(vbias.zipWithIndex.map {case (v, i) => (i, v)}, p)

        // update the states of the hidden units using the logistic activation rule
        // collect positive hidden prob and stat: array( (index, (prob, stat)) )
        val indexHiddenProbStats = weightsWithHiddenIndex.join(hbiasWithIndex).map {
          case (index, (weight, hb)) =>
            (index, updateHiddenStat(globalData.value, weight, hb, d))
        }.collect().sortBy(_._1)

        // broadcast positive hidden stats
        val globalPosHiddenStates = sc.broadcast(indexHiddenProbStats.map(_._2._2))

        // reconstruct the visible units in a similar manner
        // collect negative visible prob and stat: array( (index, (prob, stat))
        val indexVisibleProbStats = weightsWithVisibleIndex.join(vbiasWithIndex).map {
          case (index, (weight, vb)) =>
            (index, updateVisibleStat(globalPosHiddenStates.value, weight, vb, d))
        }.collect().sortBy(_._1)

        // broadcast negative visible stats
        val globalNegVisibleStats = sc.broadcast(indexVisibleProbStats.map(_._2._2))

        // update the hidden units again
        val indexHiddenProbStats2 = weightsWithHiddenIndex.join(hbiasWithIndex).map {
          case (index, (weight, hb)) =>
            (index, updateHiddenStat2(globalNegVisibleStats.value, weight, hb, d))
        }.collect().sortBy(_._1)

        // update the weights and bias which are the member of the class
        updateWeightsAndBias(data,
          indexHiddenProbStats.map(_._2._1),
          indexHiddenProbStats2.map(_._2._1),
          indexVisibleProbStats.map(_._2._2),
          learningRate,
          d)

        d += 1
      }
    }
    new RbmModel(weights, vbias, hbias)
  }

  private def updateWeightsAndBias(input: Array[Array[Int]],
                                   posHiddenProbs: Array[Double],
                                   negHiddenProbs: Array[Double],
                                   negVisibleStats: Array[Int],
                                   learningRate: Double,
                                   d: Int) {
    // w_{ij} = w_{ij} + L * (Positive(e_{ij}) - Negative(e_{ij}))
    for (i <- 0 until nVisible) {
      for (j <- 0 until nHidden) {
        // See "A Practical Guide to Training Restricted Boltzmann Machines" 3.3
        // Here we use the activation probabilities of the stats, not the stats themselves, when computing associations?
        weights(i)(j) += learningRate * (posHiddenProbs(j) * input(d)(i) - negHiddenProbs(j) * negVisibleStats(i))
      }
      vbias(i) += learningRate * (input(d)(i) - negVisibleStats(i))
    }
    for (j <- 0 until nHidden) hbias(j) += learningRate * (posHiddenProbs(j) - negHiddenProbs(j))
  }


  private def updateVisibleStat(posHiddenStates: Array[Int], weight: Array[Double], vb: Double, d: Int): (Double, Int) = {
    // compute P(v_2i=1|h_d) = f(a_i+sum_j(w_ij*h_dj)
    var negVisibleActivation = vb
    for (j <- 0 until nHidden) {
      negVisibleActivation += weight(j) * posHiddenStates(j)
    }
    val negVisibleProb = sigmoid(negVisibleActivation)
    // Sample v[0,1] from P(v_2i|h_d)
    val negVisibleStat = if (negVisibleProb > Random.nextDouble()) 1 else 0
    (negVisibleProb, negVisibleStat)
  }


  private def updateHiddenStat(input: Array[Array[Int]], weight: Array[Double], hb: Double, d: Int): (Double, Int) = {
    // compute P(h_1j=1|v_1) = f(b_j+sum_i(v_1i*w_ij))
    var posHiddenActivation = hb
    // nVisible is value on driver
    for (i <- 0 until nVisible) {
      // d is value on driver
      posHiddenActivation += input(d)(i) * weight(i)
    }
    val posHiddenProb = sigmoid(posHiddenActivation)
    // Sample h[0,1] from P(h_1j|v_d)
    val posHiddenState = if (posHiddenProb > Random.nextDouble()) 1 else 0
    (posHiddenProb, posHiddenState)
  }


  // TODO: combine with func updataHiddenStat
  private def updateHiddenStat2(negVisibleStats: Array[Int], weight: Array[Double], hb: Double, d: Int): (Double, Int) = {
    // compute P(h_1j=1|v_1) = f(b_j+sum_i(v_1i*w_ij))
    var posHiddenActivation = hb
    // nVisible is value on driver
    for (i <- 0 until nVisible) {
      // d is value on driver
      posHiddenActivation += negVisibleStats(i) * weight(i)
    }
    val posHiddenProb = sigmoid(posHiddenActivation)
    // Sample h[0,1] from P(h_1j|v_d)
    val posHiddenState = if (posHiddenProb > Random.nextDouble()) 1 else 0
    (posHiddenProb, posHiddenState)
  }


  private def sigmoid(x: Double): Double = {
    1.0 / ( 1 + math.exp(-x))
  }

}


object Rbm {

  def train(
             sc: SparkContext,
             input: Array[Array[Int]],
             numIterations: Int,
             nVisible: Int,
             nHidden: Int,
             learningRate: Double,
             p: Int,
             miniBatch: Int): RbmModel = {
    new RbmWithCD(nVisible, nHidden, learningRate)
      .run(sc, input, numIterations, p, miniBatch)
  }

  def main(args: Array[String]) {
    val test = Array(Array(1,1,1,0,0,0), Array(1,0,1,0,0,0), Array(1,1,1,0,0,0), Array(0,0,1,1,1,0), Array(0,0,1,1,0,0), Array(0,0,1,1,1,0))

    val conf = new SparkConf().setAppName("rbm").setMaster("local")
    val sc = new SparkContext(conf)
    val m = train(sc, test, 1, 6, 3, 0.1, 1, 1)
    println {m.weights.map(t => t.mkString(" ")).mkString("\n") }
  }

}