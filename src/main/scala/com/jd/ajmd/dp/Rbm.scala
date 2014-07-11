package com.jd.ajmd.dp

import breeze.linalg._
import org.apache.spark.Logging
//import breeze.numerics.sigmoid
import scala.util.Random

/**
 * Created by zhengchen on 14-7-10.
 */

class RbmModel (
  val weights: Array[Array[Double]],
  val vbias: Array[Double],
  val hbias: Array[Double]) extends Serializable


class RbmWithSGD (
  nVisible: Int,
  nHidden: Int,
  learningRate: Double) extends Serializable with Logging{

  /*
  Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  sampled from uniform distribution.
   */
  /*val weights = DenseMatrix.rand(nVisible, nHidden)
  val hbias = DenseVector.zeros[Double](nHidden)
  val vbias = DenseVector.zeros[Double](nVisible)*/

  val weights = new Array[Array[Double]](nVisible)
  for (i <- 0 to nVisible-1) weights(i) = Array.fill(nHidden)(0.1 * Random.nextDouble())
  val hbias = Array.fill(nHidden)(0.1 * Random.nextDouble())
  val vbias = Array.fill(nVisible)(0.1 * Random.nextDouble())
  //println(weights.map(t => t.mkString(" ")).mkString(","))

  def run(input: Array[Array[Int]], numIterations: Int): RbmModel = {

    for (t <- 1 to numIterations) {

      for (d <- 0 until input.length) {

        val posHiddenStates = new Array[Int](nHidden)
        val posHiddenProbs = new Array[Double](nHidden)
        for (j <- 0 until nHidden) {
          // compute P(h_1j=1|v_1) = f(b_j+sum_i(v_1i*w_ij))
          var posHiddenActivation = hbias(j)
          for (i <- 0 until nVisible) {
            posHiddenActivation += input(d)(i) * weights(i)(j)
          }
          posHiddenProbs(j) = sigmoid(posHiddenActivation)
          // Sample h[0,1] from P(h_1j|v_d)
          posHiddenStates(j) = if (posHiddenProbs(j) > Random.nextDouble()) 1 else 0
        }

        val negVisibleStates = new Array[Int](nVisible)
        val negVisibleProbs = new Array[Double](nVisible)
        for (i <- 0 until nVisible) {
          // compute P(v_2i=1|h_d) = f(a_i+sum_j(w_ij*h_dj)
          var negVisibleActivation = vbias(i)
          for (j <- 0 until nHidden) {
            negVisibleActivation += weights(i)(j) * posHiddenStates(j)
          }
          negVisibleProbs(i) = sigmoid(negVisibleActivation)
          // Sample v[0,1] from P(v_2i|h_d)
          negVisibleStates(i) = if (negVisibleProbs(i) > Random.nextDouble()) 1 else 0
        }

        val negHiddenProbs = new Array[Double](nHidden)
        for (j <- 0 until nHidden) {
          // compute P(h_2j=1|v_2) = f(b_j + sum_j(v_2i*w_ij))
          var negHiddenActivation = 0.0
          for (i <- 0 until nVisible) {
            negHiddenActivation += negVisibleStates(i) * weights(i)(j)
          }
          negHiddenProbs(j) = sigmoid(negHiddenActivation)
        }

        // update
        for (i <- 0 until nVisible) {
          for (j <- 0 until nHidden) {
            // or use negVisbleProbs
            weights(i)(j) += learningRate * (posHiddenProbs(j) * input(d)(i) - negHiddenProbs(j) * negVisibleStates(i))
          }
          vbias(i) += learningRate * (input(d)(i) - negVisibleStates(i))
        }
        for (j <- 0 until nHidden) hbias(j) += learningRate * (posHiddenProbs(j) - negHiddenProbs(j))
      }
    }
    new RbmModel(weights, vbias, hbias)

  }

  private def sigmoid(x: Double): Double = {
    1.0 / ( 1 + math.exp(-x))
  }

}

object Rbm {

  def train(
    input: Array[Array[Int]],
    numIterations: Int,
    nVisible: Int,
    nHidden: Int,
    learningRate: Double): RbmModel = {
      new RbmWithSGD(nVisible, nHidden, learningRate)
        .run(input, numIterations)
  }

  def main(args: Array[String]) {
    val test = Array(Array(1,1,1,0,0,0), Array(1,0,1,0,0,0), Array(1,1,1,0,0,0), Array(0,0,1,1,1,0), Array(0,0,1,1,0,0), Array(0,0,1,1,1,0))
    val m = train(test, 1, 6, 3, 0.1)
    println {m.weights.map(t => t.mkString(" ")).mkString("\n") }
  }

}
