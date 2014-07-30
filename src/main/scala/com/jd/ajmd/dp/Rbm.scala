package com.jd.ajmd.dp

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import scala.math.{log => mathLog, _}
import scala.collection.mutable.{ArrayBuffer, ListBuffer, HashMap, LinkedList}
import com.jd.ajmd.util.MathUtil
import scala.math.pow
import scala.collection.mutable

/**
 * Created by zhengchen on 14-7-10.
 */

class RbmWithCD (
  nVisible: Int,
  nHidden: Int,
  nRating: Int,
  learningRate: Double,
  nData: Int) extends Serializable with Logging {

  val weights = new Array[Array[Array[Double]]](nVisible)
  val vbias = new Array[Array[Double]](nVisible)
  val hbias = new Array[Double](nHidden)


  private def initWeightsAndBias(data: RDD[Map[Int, Int]]): Int = {
    // "A practical guide to training restricted boltzmann machines" 8
    val gau = new Gaussian(0.0, 0.01)
    for (i <- 0 until nVisible) {
      weights(i) = new Array[Array[Double]](nHidden)
      for (j <- 0 until nHidden) {
        //weights(i)(j) = Array.fill(nRating)(0.0)
        weights(i)(j) = Array.fill(nRating)(gau.sample())
      }
      vbias(i) = Array.fill(nRating)(gau.sample())
      //vbias(i) = Array.fill(nRating)(0.0)
    }
    //TODO: How to init the visible bias in cf-rbm?
    // visible bias: log[p_i/(1-p_i)] where p_i is the proportion of training vector in which unit i is on.
    /*val pmap = data.flatMap(_.toSeq).countByValue()
    val total = pmap.size
    pmap.foreach( pair => vbias(pair._1._1)(pair._1._2) = mathLog(pair._2.toDouble/(total-pair._2)))*/

    // hidden bias: Not using a sparsity target.
    for (j <- 0 until nHidden) hbias(j) = 0.0
    data.count().toInt
  }


  /**
   * Run RBM with cd.
   *
   * @param sc - spark context
   * @param data - input data
   * @param numIterations - max num of iterations
   * @param p - num of partitions
   *
   * @return a RbmModel with edge-weights and bias.
   */
  def run(sc: SparkContext, data: RDD[Map[Int, Int]], numIterations: Int, p: Int): RbmWithCD = {

    val nUser = initWeightsAndBias(data)
    var globalWeights = sc.broadcast(weights)
    var globalHbias = sc.broadcast(hbias)
    var globalVbias = sc.broadcast(vbias)
    // "Restricted Boltzmann Machines for Collaborative Filtering" 2
    // The full gradients with respect to the shared weight parameters can then be obtained by averaging over all N users.
    // TODO: What`s the meaning of averaging?
    val cw = data.flatMap(_.toSeq).map(_._1).countByValue().toMap

    val rmseArr = new ArrayBuffer[Double](numIterations)
    var t = 1
    while (t <= numIterations) {
      //val w = data.sample(0.01).mapPartitions(it => {
      val w = data.mapPartitions(it => {
        //TODO: replace the gradient init in partition by using accumulable or something else?
        val gradient = Array.fill(nVisible, nHidden, nRating)(0.0)
        val hbGradient = Array.fill(nHidden)(0.0)
        val vbGradient = Array.fill(nVisible, nRating)(0.0)

        while (it.hasNext) {
          computeGradient(it.next(), globalHbias.value, globalVbias.value, globalWeights.value, gradient, hbGradient, vbGradient)
        }
        Iterator.single((gradient, vbGradient, hbGradient))
      }).reduce(addTuple3)

      updateWeightsAndBias(w._1, w._2, w._3, learningRate, nData, t, cw)
      //TODO: The weights is too large to broadcast every time to be updated!
      globalWeights = sc.broadcast(weights)
      globalHbias = sc.broadcast(hbias)
      globalVbias = sc.broadcast(vbias)

      if ((t-1) % 10 == 0) {
        rmseArr.append(computeRmse(data, globalWeights.value, globalVbias.value, globalHbias.value, nData))
        println("TRAIN RMSE: "+rmseArr((t-1)/10))
      }
      t += 1
    }
    rmseArr.append(computeRmse(data, globalWeights.value, globalVbias.value, globalHbias.value, nData))
    println(rmseArr.mkString("\n"))
    this
  }


  private def computeGradient(input: Map[Int, Int],
                              hBia: Array[Double],
                              vBia: Array[Array[Double]],
                              weight: Array[Array[Array[Double]]],
                              gradient: Array[Array[Array[Double]]],
                              hbGradient: Array[Double],
                              vbGradient: Array[Array[Double]]) = {

    val (posHiddenProbs, posHiddenStates) = runHidden(input, weight, hBia)
    val (negVisibleProbs, negVisibleStats) = runVisible(input, weight, vBia, posHiddenStates)
    val (negHiddenProbs, negHiddenStates) = runHidden(negVisibleStats, weight, hBia)

    input.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          gradient(k)(j)(v) += posHiddenStates(j)
        }
        vbGradient(k)(v) += 1.0
      }
    }
    negVisibleStats.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          gradient(k)(j)(v) -= negHiddenStates(j)
        }
        vbGradient(k)(v) -= 1.0
      }
    }
    for (j <- 0 until nHidden) hbGradient(j) += posHiddenStates(j) - negHiddenStates(j)

  }


  private def runHidden(input: Map[Int, Int],
                         weight: Array[Array[Array[Double]]],
                         hBia: Array[Double]): (Array[Double], Array[Int]) = {
    val posHiddenActivations = hBia.clone()
    for (j <- 0 until nHidden) {
      input.foreach {
        case(k, v) => {
          posHiddenActivations(j) += weight(k)(j)(v)
        }
      }
    }
    val posHiddenProbs = posHiddenActivations.map(sigmoid(_))
    val posHiddenStates = posHiddenProbs.map(MathUtil.binomial(_))
    (posHiddenProbs, posHiddenStates)
  }


  private def runVisible(input: Map[Int, Int],
                          weight: Array[Array[Array[Double]]],
                          vBia: Array[Array[Double]],
                          posHiddenStates: Array[Int]): (Map[Int, Array[Double]], Map[Int, Int]) = {
    val negVisibleActivations = input.map{ case(k, v) => (k, vBia(k).clone()) }.toMap
    input.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          for (kr <- 0 until nRating) {
            negVisibleActivations(k)(kr) += weight(k)(j)(kr) * posHiddenStates(j)
          }
        }
      }
    }
    val negVisibleProbs = negVisibleActivations.map { case(k, v) => (k ,MathUtil.softMax(v)) }.toMap
    val negVisibleStats = negVisibleProbs.map { case(k, v) => (k, MathUtil.multinomial(v))}.toMap
    (negVisibleProbs, negVisibleStats)
  }


  private def updateWeightsAndBias(
                                   wg: Array[Array[Array[Double]]],
                                   vbg: Array[Array[Double]],
                                   hbg: Array[Double],
                                   learningRate: Double,
                                   numData: Int,
                                   nIter: Int,
                                   cw: Map[Int, Long]) {
    val thisIterStepSize = learningRate / math.sqrt(nIter)
    for (j <- 0 until nHidden) {
      for (i <- 0 until nVisible) {
        for (k <- 0 until nRating) {
          if(cw.contains(i))
            weights(i)(j)(k) += thisIterStepSize * (wg(i)(j)(k)/cw(i))
        }
      }
      hbias(j) += thisIterStepSize * (hbg(j)/numData)
    }
    for (i <- 0 until nVisible) {
      for (k <- 0 until nRating) {
        if(cw.contains(i))
          vbias(i)(k) += thisIterStepSize * (vbg(i)(k)/cw(i))
      }
    }
  }


  private def addTuple3(left: (Array[Array[Array[Double]]], Array[Array[Double]], Array[Double]),
                        right: (Array[Array[Array[Double]]], Array[Array[Double]], Array[Double])
                         ): (Array[Array[Array[Double]]], Array[Array[Double]], Array[Double]) = {

    def addWeight(left: Array[Array[Array[Double]]], right: Array[Array[Array[Double]]]): Array[Array[Array[Double]]] = {
      val result = new Array[Array[Array[Double]]](nVisible)
      for (i <- 0 until nVisible) {
        for (j <- 0 until nHidden) {
          for (k <- 0 until nRating) {
            result(i)(j)(k) = left(i)(j)(k) + right(i)(j)(k)
          }
        }
      }
      result
    }

    def addHBias(left: Array[Double], right: Array[Double]): Array[Double] = {
      val result = new Array[Double](nHidden)
      for (j <- 0 until nHidden) {
        result(j) = left(j) + right(j)
      }
      result
    }

    def addVBias(left: Array[Array[Double]], right: Array[Array[Double]]): Array[Array[Double]] = {
      val result = new Array[Array[Double]](nVisible)
      for (i <- 0 until nVisible) {
        for (k <- 0 until nRating) {
          result(i)(k) = left(i)(k) + right(i)(k)
        }
      }
      result
    }

    (addWeight(left._1, right._1), addVBias(left._2, right._2), addHBias(left._3, right._3))
  }


  def computeRmse(data: RDD[Map[Int, Int]],
                  weight: Array[Array[Array[Double]]],
                  vBia: Array[Array[Double]],
                  hBia: Array[Double],
                  nData: Int): Double = {
    val tErr = data.map {
      input => {
        val (posHiddenProbs, posHiddenStates) = runHidden(input, weight, hBia)
        val (negVisibleProbs, negVisibleStats) = runVisible(input, weight, vBia, posHiddenStates)
        var err = 0.0
        input.foreach {
          case(k, v) => {
            val a = negVisibleProbs(k)
            val predict = MathUtil.expectSoftMax(a)
            //println(v+" "+predict)
            err += pow(v-predict, 2)
          }
        }
        err
      }
    }.reduce(_+_)
    math.sqrt(tErr/nData)
  }

  def predict(train: RDD[(Int, Map[Int, Int])], test: RDD[(Int, Map[Int, Int])],
              weight: Array[Array[Array[Double]]],
              vBia: Array[Array[Double]],
              hBia: Array[Double],
              nData: Int): Double = {
    val tErr = train.join(test).map {
      case(k, (trn, tst)) => {
        val (posHiddenProbs, posHiddenStates) = runHidden(trn, weight, hBia)
        val (negVisibleProbs, negVisibleStats) = runVisible(tst, weight, vBia, posHiddenStates)
        var err = 0.0
        tst.foreach {
          case(k, v) => {
            val a = negVisibleProbs(k)
            val predict = MathUtil.expectSoftMax(a)
            err += pow(v-predict, 2.0)
          }
        }
        err
      }
    }.reduce(_+_)
    math.sqrt(tErr/nData)
  }

}


object Rbm {

  def train(
             sc: SparkContext,
             input: RDD[Map[Int, Int]],
             numIterations: Int,
             nVisible: Int,
             nHidden: Int,
             nRating: Int,
             learningRate: Double,
             nData: Int,
             p: Int): RbmWithCD = {
    new RbmWithCD(nVisible, nHidden, nRating, learningRate, nData)
      .run(sc, input, numIterations, p)
  }

  // TODO: use fastutil to replace map. http://fastutil.di.unimi.it/
  def loadData(fileData: RDD[String]): RDD[(Int, Map[Int, Int])] = {
    fileData.map {
      line => {
        val cons = line.split(" ")
        (cons(0).toInt-1, Map(cons(1).toInt-1 -> (cons(2).toInt-1)))
      }
    }.reduceByKey(_ ++ _)
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("rbm").setMaster("local")
    val sc = new SparkContext(conf)
    val trainData = sc.textFile("file://d:/Work/RBM/data/train.txt").cache()
    //val trainData = sc.textFile("file://d:/Work/RBM/data/train_toy.txt").cache()
    val trainSize = trainData.count().toInt
    val itemSize = trainData.map(_.split(" ")(1)).distinct().count().toInt
    val trainInput = loadData(trainData).map(_._2).cache()
    trainData.unpersist()

    val numParallel = 1
    //val model = train(sc, trainInput, 1, itemSize, 2, 5, 0.01, trainSize, numParallel)
    val model = train(sc, trainInput, 200, 1682, 50, 5, 0.1, trainSize, numParallel)

    val testData = sc.textFile("file://d:/Work/RBM/data/test.txt").cache()
    val testSize = testData.count().toInt
    val gWeights = sc.broadcast(model.weights)
    val gVbias = sc.broadcast(model.vbias)
    val gHbias = sc.broadcast(model.hbias)
    println("TEST RMSE: " + model.predict(loadData(trainData), loadData(testData),
      gWeights.value, gVbias.value, gHbias.value, testSize))

  }

}