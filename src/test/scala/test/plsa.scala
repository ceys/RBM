package test

import scala.math.random

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object plsa {
	def run(data: RDD[Array[Int]], numTopics: Int, numWords: Int, maxIter: Int) = {
	  var dataPartitions = data.mapPartitions(docs =>{
	    var wz = new Array[Array[Double]](numTopics)
	    for (i <- 0 until numTopics) wz(i) = new Array[Double](numWords)
	    val docsWithdz = docs.toArray.map(doc => 
	      (doc, (new Array[Double](numTopics)).map(_ => 1.0 / numTopics)))
	    Iterator.single((docsWithdz, wz))
	  })
	  
	  var wzMatrix = new Array[Array[Double]](numTopics)
	  for (i <- 0 until numTopics) {
	    wzMatrix(i) = (new Array[Double](numWords)).map(p => p + random)
	    val sum = wzMatrix(i).sum
	    wzMatrix(i) = wzMatrix(i).map(p => p / sum)
	  }
	  //printMatrix(wzMatrix, "wz")
	  
	  println(wzMatrix.length)
	  println(wzMatrix.head.length)
	  
	  val sc = data.context
	  for (i <- 0 until maxIter){
	    var globalWZ = sc.broadcast(wzMatrix)
	    val newData = dataPartitions.mapPartitions({it => 
		  var (docsWithdz, wz) = it.next
		  for (i <- 0 until wz.length) wz(i).map(p => 0.0)
		  //println(docsWithdz.length)
		  for(j <- 0 until docsWithdz.length){
		    var (doc, dz) = docsWithdz(j)
		    //printVector(doc,"doc")
		    //printVector(dz, "dz")
		  }
		  computeQ(docsWithdz, wz, globalWZ.value)
		  for (d <- 0 until docsWithdz.length){
		    //printVector(docsWithdz(d)._1, "doc after update")
		    //printVector(docsWithdz(d)._2, "dz after update")
		  }
		  Iterator.single(docsWithdz, wz)
		}, true).persist
		
		if (i < 20) dataPartitions = newData
		//var newwzMatrix = dataPartitions.map(_._2).reduce((A, B) => plus(A, B))
		wzMatrix = wzMatrix.map(oneTopic => {
		  val sum = oneTopic.sum
		  oneTopic.map(word => word / sum)})
		//priviousData.unpersist(false)
		//dataPartitions.checkpoint
		//globalWZ.unpersist
		//printMatrix(wzMatrix, "wz")
	  }
	  //printMatrix(wzMatrix, "wz")
	}
	
	def computeQ(
	    docs: Array[(Array[Int], Array[Double])],
	    wzPart: Array[Array[Double]],
	    wzGlobal: Array[Array[Double]]) = {
	  val numTopics = wzPart.length
	  var Q = new Array[Double](numTopics)
	  for(d <- 0 until docs.length){
	    var (doc, dz) = docs(d)
	    var newDz = new Array[Double](numTopics)
	    for (i <- 0 until doc.length){
	      for(j <- 0 until numTopics){
	        Q(j) = wzGlobal(j)(doc(i)) * dz(j)
	      }
	      val sum = Q.sum
	      Q = Q.map(_/sum)
	      //printVector(Q, "Q")
	      for (k <- 0 until numTopics) {
	        newDz(k) += Q(k)
	        wzPart(k)(doc(i)) += Q(k)
	      }
	    }
	    
	    val sum = newDz.sum
	    docs(d) = (doc,newDz.map(p => p / sum))
	    //printVector(Q, "newDz")
	  }
	}
	
	private def plus(matrixA: Array[Array[Double]], matrixB: Array[Array[Double]])
      : Array[Array[Double]] = {
		for (i <- 0 until matrixA.length)
			for(j <- 0 until matrixA.head.length)
				matrixA(i)(j) += matrixB(i)(j)
		matrixA
	}
	
	def printVector(v: Array[Double], name: String) = {
		var str = new String(name)
		for (i <- 0 until v.length){
			str += "  " + v(i)
		}
		println(str)
	}
	
	def printVector(v: Array[Int], name: String) = {
		var str = new String(name)
		for (i <- 0 until v.length){
			str += "  " + v(i)
		}
		println(str)
	}
	
	def printMatrix(m: Array[Array[Double]], name: String) = {
		println(name)
		for (i <- 0 until m.length){
			var str = "class " + i      
			for (j <- 0 until m(i).length){
				str += "  " + m(i)(j)
			}
		println(str)
		}
	}
	
	def main(args: Array[String]){
		println("plsa start")
		val numTopics = 10
		val numWords = 15
		val maxIter = 500
		
		val checkPointDir = System.getProperty("spark.gibbsSampling.checkPointDir", "D:\\")
		val conf = new SparkConf().setAppName("testSoftmax").setMaster("local")
		val spark = new SparkContext(conf)
		spark.setCheckpointDir(checkPointDir)

		//var docs = spark.textFile("D:\\unfold.dat", 1).map(doc => doc.split(" "))
		//var docsRDD = docs.map({ids => ids.map(id => id.toInt)})
		//val count = docsRDD.flatMap(ids => ids).distinct.count
		//println(count)
		
		var docs = new Array[Array[Int]](15)
		docs(0) = Array(1,2,3,4)
		docs(1) = Array(2,3,4,0)
		docs(2) = Array(1,0,3,4)
		docs(3) = Array(1,2,0,4)
		docs(4) = Array(1,2,3,0)
		docs(5) = Array(5,6,7,8)
		docs(6) = Array(6,7,8,9)
		docs(7) = Array(5,7,8,9)
		docs(8) = Array(5,6,8,9)
		docs(9) = Array(5,6,7,9)
		docs(10) = Array(10,11,12,13)
		docs(11) = Array(11,12,13,14)
		docs(12) = Array(12,13,14,10)
		docs(13) = Array(13,14,10,11)
		docs(14) = Array(14,10,11,12)
		val docsRDD = spark.parallelize(docs, 1)
    val count = docsRDD.flatMap(ids => ids).distinct.count

		run(docsRDD, numTopics, count.toInt, maxIter)
	}
}