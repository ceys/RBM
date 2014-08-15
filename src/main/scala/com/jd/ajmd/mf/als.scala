package com.jd.ajmd.mf

import com.esotericsoftware.kryo.Kryo
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.{KryoSerializer, KryoRegistrator}
import scala.collection.mutable.PriorityQueue
import org.jblas.DoubleMatrix

/**
 * Created by zhengchen on 14-8-14.
 */
object Als {

  class ALSRegistrator extends KryoRegistrator {
    override def registerClasses(kryo: Kryo) {
      kryo.register(classOf[Rating])
    }
  }

  case class Params(
                     input: String = null,
                     kryo: Boolean = false,
                     numIterations: Int = 20,
                     lambda: Double = 1.0,
                     rank: Int = 10,
                     implicitPrefs: Boolean = false,
                     blocks: Int = 1,
                     alpha: Double = 40.0,
                     topn: Int = 10,
                     output: String = null,
                     master: String = "local")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("MovieLensALS") {
      head("MovieLensALS: an example app for ALS on MovieLens data.")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      opt[Unit]("kryo")
        .text(s"use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Unit]("implicitPrefs")
        .text("use implicit preference")
        .action((_, c) => c.copy(implicitPrefs = true))
      opt[String]("master")
        .text("local yarn-client or yarn-cluster")
        .action((x, c) => c.copy(master = x))
      opt[Int]("blocks")
        .text(s"How many partitions to use in the resulting RDD")
        .action((x, c) => c.copy(blocks = x))
      arg[String]("<input>")
        .required()
        .text("input paths to a MovieLens dataset of ratings")
        .action((x, c) => c.copy(input = x))
      opt[String]("output")
        .text("output paths to a recommendations of users")
        .action((x, c) => c.copy(output = x))
      opt[Int]("topn")
        .text("the number of recommendations per user")
        .action((x, c) => c.copy(topn = x))
      opt[Double]("alpha")
        .text(s"used in computing confidence in implicit ALS, default is  ${defaultParams.alpha}")
        .action((x, c) => c.copy(alpha = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.MovieLensALS \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --rank 5 --numIterations 20 --lambda 1.0 --kryo \
          |  data/mllib/sample_movielens_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"MovieLensALS with $params")
                              .setMaster(params.master)
    if (params.kryo) {
      conf.set("spark.serializer", classOf[KryoSerializer].getName)
        .set("spark.kryo.registrator", classOf[ALSRegistrator].getName)
        .set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val ratings = sc.textFile(params.input).map { line =>
      val fields = line.split(" ")
      if (params.implicitPrefs) {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      } else {
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }
    }.cache()

    val numRatings = ratings.count()
    val numUsers = ratings.map(_.user).distinct().count()
    val numMovies = ratings.map(_.product).distinct().count()

    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

    val splits = ratings.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = if (params.implicitPrefs) {
      /*
       * 0 means "don't know" and positive values mean "confident that the prediction should be 1".
       * Negative values means "confident that the prediction should be 0".
       * We have in this case used some kind of weighted RMSE. The weight is the absolute value of
       * the confidence. The error is the difference between prediction and either 1 or 0,
       * depending on whether r is positive or negative.
       */
      splits(1).map(x => Rating(x.user, x.product, if (x.rating > 0) 1.0 else 0.0))
    } else {
      splits(1)
    }.cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    ratings.unpersist(blocking = false)

    val model = new ALS()
      .setRank(params.rank)
      .setIterations(params.numIterations)
      .setLambda(params.lambda)
      .setImplicitPrefs(params.implicitPrefs)
      .setBlocks(params.blocks)
      .setAlpha(params.alpha)
      .run(training)

    val trainRmse = computeRmse(model, training, params.implicitPrefs)
    val rmse = computeRmse(model, test, params.implicitPrefs)

    //val userProducts = ratings.map( x => (x.user, x.product) ).groupByKey(params.blocks).map( x => (x._1, x._2.toSet))
    //predictAll(model, userProducts, params.topn).saveAsTextFile(params.output)
    println(s"Training RMSE = $trainRmse.")
    println(s"Test RMSE = $rmse.")

    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean) = {

    def mapPredictedRating(r: Double) = if (implicitPrefs) math.max(math.min(r, 1.0), 0.0) else r

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), mapPredictedRating(x.rating))
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def predictAll(model: MatrixFactorizationModel, userProducts: RDD[(Int, Set[Int])], k: Int): RDD[String] = {

    val sc = model.productFeatures.context
    val pfs = sc.broadcast(model.productFeatures.collect()).value
    userProducts.join(model.userFeatures).map {
      case (user, (uProducts, uFeatures)) => {
        val topNProducts = new PriorityQueue[(Int, Double)]()(Ordering.by(t => -t._2))
        topNProducts ++= Array.fill(k)((0,0.0))

        val userVector = new DoubleMatrix(uFeatures)
        pfs.map {
          case (product, pFeatures) => {
            if (!uProducts.contains(product)) {
              val productVector = new DoubleMatrix(pFeatures)
              val rating = userVector.dot(productVector)
              if (topNProducts.head._2 < rating) {
                topNProducts += ((product, rating))
                topNProducts.dequeue()
              }
            }
          }
        }
        user.toString + "\t" + topNProducts.map(x => x._1.toString + "," + x._2.toString).mkString(" ")
      }
    }
  }
}
