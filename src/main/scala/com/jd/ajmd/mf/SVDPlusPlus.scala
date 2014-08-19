package com.jd.ajmd.mf

/**
 * Created by zhengchen on 14-8-14.
 */
import scala.util.Random
import org.jblas.DoubleMatrix
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser

/** Implementation of SVD++ algorithm. */
object SVDPlusPlus {

  /** Configuration parameters for SVDPlusPlus. */
  class Conf(
              var rank: Int,
              var maxIters: Int,
              var minVal: Double,
              var maxVal: Double,
              var gamma1: Double,
              var gamma2: Double,
              var gamma6: Double,
              var gamma7: Double)
    extends Serializable

  /**
   * Implement SVD++ based on "Factorization Meets the Neighborhood:
   * a Multifaceted Collaborative Filtering Model",
   * available at [[http://public.research.att.com/~volinsky/netflix/kdd08koren.pdf]].
   *
   * The prediction rule is rui = u + bu + bi + qi*(pu + |N(u)|^(-0.5)*sum(y)),
   * see the details on page 6.
   *
   * @param edges edges for constructing the graph
   *
   * @param conf SVDPlusPlus parameters
   *
   * @return a graph with vertex attributes containing the trained model
   */
  def run(edges: RDD[Edge[Double]], conf: Conf)
  : (Graph[(DoubleMatrix, DoubleMatrix, Double, Double), Double], Double) =
  {
    // Generate default vertex attribute
    def defaultF(rank: Int): (DoubleMatrix, DoubleMatrix, Double, Double) = {
      val v1 = new DoubleMatrix(rank)
      val v2 = new DoubleMatrix(rank)
      for (i <- 0 until rank) {
        v1.put(i, Random.nextDouble())
        v2.put(i, Random.nextDouble())
      }
      (v1, v2, 0.0, 0.0)
    }

    // calculate global rating mean
    edges.cache()
    val (rs, rc) = edges.map(e => (e.attr, 1L)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val u = rs / rc

    // construct graph
    var g = Graph.fromEdges(edges, defaultF(conf.rank)).cache()

    // Calculate initial bias and norm
    val t0 = g.mapReduceTriplets(
      et => Iterator((et.srcId, (1L, et.attr)), (et.dstId, (1L, et.attr))),
      (g1: (Long, Double), g2: (Long, Double)) => (g1._1 + g2._1, g1._2 + g2._2))

    g = g.outerJoinVertices(t0) {
      (vid: VertexId, vd: (DoubleMatrix, DoubleMatrix, Double, Double),
       msg: Option[(Long, Double)]) =>
        (vd._1, vd._2, msg.get._2 / msg.get._1, 1.0 / scala.math.sqrt(msg.get._1))
    }

    def mapTrainF(conf: Conf, u: Double)
                 (et: EdgeTriplet[(DoubleMatrix, DoubleMatrix, Double, Double), Double])
    : Iterator[(VertexId, (DoubleMatrix, DoubleMatrix, Double))] = {
      val (usr, itm) = (et.srcAttr, et.dstAttr)
      val (p, q) = (usr._1, itm._1)
      var pred = u + usr._3 + itm._3 + q.dot(usr._2)
      pred = math.max(pred, conf.minVal)
      pred = math.min(pred, conf.maxVal)
      val err = et.attr - pred
      val updateP = q.mul(err)
        .subColumnVector(p.mul(conf.gamma7))
        .mul(conf.gamma2)
      val updateQ = usr._2.mul(err)
        .subColumnVector(q.mul(conf.gamma7))
        .mul(conf.gamma2)
      val updateY = q.mul(err * usr._4)
        .subColumnVector(itm._2.mul(conf.gamma7))
        .mul(conf.gamma2)
      Iterator((et.srcId, (updateP, updateY, (err - conf.gamma6 * usr._3) * conf.gamma1)),
        (et.dstId, (updateQ, updateY, (err - conf.gamma6 * itm._3) * conf.gamma1)))
    }

    for (i <- 0 until conf.maxIters) {
      // Phase 1, calculate pu + |N(u)|^(-0.5)*sum(y) for user nodes
      g.cache()
      val t1 = g.mapReduceTriplets(
        et => Iterator((et.srcId, et.dstAttr._2)),
        (g1: DoubleMatrix, g2: DoubleMatrix) => g1.addColumnVector(g2))
      g = g.outerJoinVertices(t1) {
        (vid: VertexId, vd: (DoubleMatrix, DoubleMatrix, Double, Double),
         msg: Option[DoubleMatrix]) =>
          if (msg.isDefined) (vd._1, vd._1
            .addColumnVector(msg.get.mul(vd._4)), vd._3, vd._4) else vd
      }

      // Phase 2, update p for user nodes and q, y for item nodes
      g.cache()
      val t2 = g.mapReduceTriplets(
        mapTrainF(conf, u),
        (g1: (DoubleMatrix, DoubleMatrix, Double), g2: (DoubleMatrix, DoubleMatrix, Double)) =>
          (g1._1.addColumnVector(g2._1), g1._2.addColumnVector(g2._2), g1._3 + g2._3))
      g = g.outerJoinVertices(t2) {
        (vid: VertexId,
         vd: (DoubleMatrix, DoubleMatrix, Double, Double),
         msg: Option[(DoubleMatrix, DoubleMatrix, Double)]) =>
          (vd._1.addColumnVector(msg.get._1), vd._2.addColumnVector(msg.get._2),
            vd._3 + msg.get._3, vd._4)
      }
    }

    // calculate error on training set
    def mapTestF(conf: Conf, u: Double)
                (et: EdgeTriplet[(DoubleMatrix, DoubleMatrix, Double, Double), Double])
    : Iterator[(VertexId, Double)] =
    {
      val (usr, itm) = (et.srcAttr, et.dstAttr)
      val (p, q) = (usr._1, itm._1)
      var pred = u + usr._3 + itm._3 + q.dot(usr._2)
      pred = math.max(pred, conf.minVal)
      pred = math.min(pred, conf.maxVal)
      val err = (et.attr - pred) * (et.attr - pred)
      Iterator((et.dstId, err))
    }
    g.cache()
    val t3 = g.mapReduceTriplets(mapTestF(conf, u), (g1: Double, g2: Double) => g1 + g2)
    g = g.outerJoinVertices(t3) {
      (vid: VertexId, vd: (DoubleMatrix, DoubleMatrix, Double, Double), msg: Option[Double]) =>
        if (msg.isDefined) (vd._1, vd._2, vd._3, msg.get) else vd
    }

    (g, u)
  }

  case class Params(
                     rank: Int = 10,
                     maxIters: Int = 10,
                     input: String= null,
                     minVal: Double = 0.0,
                     maxVal: Double = 5.0,
                     gamma1: Double = 0.007,
                     gamma2: Double = 0.007,
                     gamma6: Double = 0.005,
                     gamma7: Double = 0.015,
                     master: String = "yarn-cluster")

  def main(args: Array[String]) {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("svd++") {
      head("svd++.")
      opt[Int]("rank")
        .text(s"rank, default: ${defaultParams.rank}}")
        .action((x, c) => c.copy(rank = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default: ${defaultParams.maxIters}")
        .action((x, c) => c.copy(maxIters = x))
      opt[Int]("minVal")
        .text(s"minVal, default: ${defaultParams.minVal}")
        .action((x, c) => c.copy(minVal = x))
      opt[Int]("maxVal")
        .text(s"maxVal, default: ${defaultParams.maxVal}")
        .action((x, c) => c.copy(maxVal = x))
      opt[Int]("gamma1")
        .text(s"gamma1, default: ${defaultParams.gamma1}")
        .action((x, c) => c.copy(gamma1 = x))
      opt[Int]("gamma2")
        .text(s"gamma2, default: ${defaultParams.gamma2}")
        .action((x, c) => c.copy(gamma2 = x))
      opt[Int]("gamma6")
        .text(s"gamma6, default: ${defaultParams.gamma6}")
        .action((x, c) => c.copy(gamma1 = x))
      opt[Int]("gamma7")
        .text(s"gamma7, default: ${defaultParams.gamma7}")
        .action((x, c) => c.copy(gamma1 = x))
      opt[String]("master")
        .text("local yarn-client or yarn-cluster")
        .action((x, c) => c.copy(master = x))
      arg[String]("<input>")
        .required()
        .text("input paths to a dataset of ratings")
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

    def run(params: Params) {
      val conf = new SparkConf().setAppName(s"Svd++ with $params")
        //.setMaster(params.master)
      val sc = new SparkContext(conf)

      val edges = sc.textFile(params.input).map { line =>
        val fields = line.split(",")
        Edge(fields(0).toLong * 2, fields(1).toLong * 2 + 1, fields(2).toDouble)
      }
      val svdConf = new SVDPlusPlus.Conf(params.rank, params.maxIters,
        params.minVal, params.maxVal, params.gamma1, params.gamma2, params.gamma6, params.gamma7)
      var (graph, u) = SVDPlusPlus.run(edges, svdConf)
      graph.cache()
      val err = graph.vertices.collect().map{ case (vid, vd) =>
        if (vid % 2 == 1) vd._4 else 0.0
      }.reduce(_ + _) / graph.triplets.collect().size
      println(err)
    }

  }
}
