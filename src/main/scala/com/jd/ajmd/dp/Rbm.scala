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

  // Initialize a weight matrix, of dimensions (num_visible x num_hidden),
  // sampled from gaussian distribution.
  // the first row and first column represent the weights of bias

  val weights = new Array[Array[Array[Double]]](nVisible)
  val vbias = new Array[Array[Double]](nVisible)
  val hbias = new Array[Double](nHidden)


  private def initWeightsAndBias(data: RDD[Map[(Int, Int), Int]]): Int = {
    // "A practical guide to training restricted boltzmann machines" 8
    val gau = new Gaussian(0.0, 0.01)
    for (i <- 0 until nVisible) {
      weights(i) = new Array[Array[Double]](nHidden)
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

    //println(vbias.deep.mkString("\n"))
    //println(hbias.deep.mkString("\n"))
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
  def run(sc: SparkContext, data: RDD[Map[(Int, Int), Int]], numIterations: Int, p: Int): RbmWithCD = {
    val nUser = initWeightsAndBias(data)
    var globalWeights = sc.broadcast(weights)
    var globalHbias = sc.broadcast(hbias)
    var globalVbias = sc.broadcast(vbias)

    val cw = data.flatMap(_.toSeq).map(_._1._1).countByValue().toMap

    val rmseArr = new ArrayBuffer[Double](numIterations)
    var t = 1
    while (t <= numIterations) {
      val w = data.mapPartitions(it => {

        //TODO: replace the gradient init in partition by using accumulable or something else.
        val gradient = Array.fill(nVisible, nHidden, nRating)(0.0)
        val hbGradient = Array.fill(nHidden)(0.0)
        val vbGradient = Array.fill(nVisible, nRating)(0.0)

        while (it.hasNext) {
          computeGradient(it.next(), globalHbias.value, globalVbias.value, globalWeights.value, gradient, hbGradient, vbGradient)
        }
        Iterator.single((gradient, vbGradient, hbGradient))
      }).reduce(addTuple3)

      updateWeightsAndBias(w._1, w._2, w._3, learningRate, nUser, t, cw)
      //TODO: The weights is too large to broadcast every time need to update it!
      globalWeights = sc.broadcast(weights)
      globalHbias = sc.broadcast(hbias)
      globalVbias = sc.broadcast(vbias)

      if ((t-1) % 10 == 0) rmseArr.append(computeRmse(data, nData))
      t += 1
    }
    rmseArr.append(computeRmse(data, nData))
    println(rmseArr.mkString("\n"))
    this
  }


  private def computeGradient(input: Map[(Int, Int), Int],
                              hBias: Array[Double],
                              vBias: Array[Array[Double]],
                              weight: Array[Array[Array[Double]]],
                              gradient: Array[Array[Array[Double]]],
                              hbGradient: Array[Double],
                              vbGradient: Array[Array[Double]]) = {

    val (posHiddenProbs, posHiddenStates) = runHidden(input, weight, hBias)

    val (negVisibleProbs, negVisibleStats) = runVisible(input, weight, vBias, posHiddenStates)

    val (negHiddenProbs, negHiddenStates) = runHidden(negVisibleStats, weight, hBias)

    //println(posHiddenProbs.mkString(","))
    //println(negVisibleActivations.toArray.map(_._2).deep.mkString("\n"))
    //println(negVisibleProbs.toArray.map(_._2).deep.mkString("\n"))
    //println(negVisibleStats.toArray.deep.mkString("\t"))

    input.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          gradient(k._1)(j)(k._2) += posHiddenStates(j) * v
        }
        vbGradient(k._1)(k._2) += v
      }
    }
    negVisibleStats.foreach {
      case(k, v) => {
        for (j <- 0 until nHidden) {
          //gradient(k._1)(j)(k._2) -= negHiddenProbs(j) * v
          gradient(k._1)(j)(k._2) -= negHiddenStates(j) * v
        }
        vbGradient(k._1)(k._2) -= v
      }

    }
    //for (j <- 0 until nHidden) hbGradient(j) += posHiddenStates(j) - negHiddenProbs(j)
    for (j <- 0 until nHidden) hbGradient(j) += posHiddenStates(j) - negHiddenStates(j)

  }


  private def runHidden(input: Map[(Int, Int), Int],
                         weight: Array[Array[Array[Double]]],
                         hBias: Array[Double]): (Array[Double], Array[Int]) = {
    val posHiddenActivations = hBias.clone()
    for (j <- 0 until nHidden) {
      input.foreach {
        case(k, v) => {
          posHiddenActivations(j) += v * weight(k._1)(j)(k._2)
        }

      }
    }
    val posHiddenProbs = posHiddenActivations.map(sigmoid(_))
    val posHiddenStates = posHiddenProbs.map(MathUtil.binomial(_))
    (posHiddenProbs, posHiddenStates)
  }


  private def runVisible(input: Map[(Int, Int), Int],
                          weight: Array[Array[Array[Double]]],
                          vBias: Array[Array[Double]],
                          posHiddenStates: Array[Int]): (Map[Int, Array[Double]], Map[(Int, Int), Int]) = {
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
    val negVisibleProbs = negVisibleActivations.map { case(k, v) => (k ,MathUtil.softMax(v)) }.toMap
    val negVisibleStats = negVisibleProbs.map { case(k, v) => ((k, MathUtil.multinomial(v)), 1)}.toMap
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


  def computeRmse(data: RDD[Map[(Int, Int), Int]],
                  nData: Int): Double = {
    val tErr = data.map {
      input => {
        val (posHiddenProbs, posHiddenStates) = runHidden(input, weights, hbias)
        val (negVisibleProbs, negVisibleStats) = runVisible(input, weights, vbias, posHiddenStates)
        var err = 0.0
        input.foreach {
          case(k, v) => {
            val a = negVisibleProbs(k._1)
            /*for (t <- 0 until a.length) {
              if (t == k._2) err += pow(v - a(t), 2.0)
              else err += pow(a(t), 2.0)
            }*/
            val predict = a.indexOf(a.max)
            err += pow(k._2-predict, 2.0)
          }
        }
        err
      }
    }.reduce(_+_)
    math.sqrt(tErr/nData)
  }

  def predict(train: RDD[Map[(Int, Int), Int]], test: RDD[Map[(Int, Int), Int]], nData: Int): Double = {
    val tErr = train.zip(test).map {
      case(trn, tst) => {
        val (posHiddenProbs, posHiddenStates) = runHidden(trn, weights, hbias)
        val (negVisibleProbs, negVisibleStats) = runVisible(tst, weights, vbias, posHiddenStates)
        var err = 0.0
        tst.foreach {
          case(k, v) => {
            val a = negVisibleProbs(k._1)
            val predict = a.indexOf(a.max)
            err += pow(k._2-predict, 2.0)
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
             input: RDD[Map[(Int, Int), Int]],
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



  def main(args: Array[String]) {
    //val test = Array(Array(1,1,1,0,0,0), Array(1,0,1,0,0,0), Array(1,1,1,0,0,0), Array(0,0,1,1,1,0), Array(0,0,1,1,0,0), Array(0,0,1,1,1,0))

    import scala.io.Source

    val test = Source.fromFile("data/movielens.txt").getLines().map(_.split(" ").map(_.toInt).toSeq).toSeq
    //val test = Source.fromFile("data/sample_movielens_data.txt").getLines().map(_.split("::").map(_.toInt).toSeq).toSeq
    // TODO: use fastutil to replace map. http://fastutil.di.unimi.it/
    val allArr = new ArrayBuffer[Map[(Int, Int), Int]]()
    var map = new ArrayBuffer[((Int, Int), Int)]()
    map += (((test(0)(1)-1, test(0)(2)-1), 1))
    val totalSize = test.length
    var userSize = 1
    for (t <- 1 until test.length) {
      if (test(t)(0) != test(t-1)(0)) {
        userSize += 1
        allArr += map.toMap
        map = new ArrayBuffer[((Int, Int), Int)]()
      }
      map += (((test(t)(1)-1, test(t)(2)-1), 1))

    }
    allArr += map.toMap

    val trainArr = new ArrayBuffer[Map[(Int, Int), Int]](userSize)
    val testArr = new ArrayBuffer[Map[(Int, Int), Int]](userSize)
    var sumTrainSize = 0
    for (t <- 0 until allArr.length) {
      val size = allArr(t).size
      val trainSize = (size * 0.8).toInt
      trainArr += allArr(t).take(trainSize)
      testArr += allArr(t).takeRight(size-trainSize)
      sumTrainSize += trainSize
    }
    //println(testArr.toArray.deep.mkString("\n"))
    val numParallel = 1

    val conf = new SparkConf().setAppName("rbm").setMaster("local")
    val sc = new SparkContext(conf)
    val input = sc.parallelize(trainArr.toSeq, numParallel)
    val model = train(sc, input, 100, 1682, 200, 5, 0.001, sumTrainSize, 1)
    //val model = train(sc, input, 500, 100, 50, 5, 0.001, sumTrainSize, 1)

    val testInput = sc.parallelize(testArr.toSeq, numParallel)

    println("TEST RMSE: " + model.predict(input, testInput, totalSize-sumTrainSize))

    // val m = train(sc, input, 800, 100, 10, 5, 0.1, numParallel)
    //println(m.weights.deep.mkString("\n"))
    //println {toArray.map(t => t.mkString(" ")).mkString("\n") }
  }

}