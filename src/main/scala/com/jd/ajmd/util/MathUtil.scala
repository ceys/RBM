package com.jd.ajmd.util

import scala.util.Random
import scala.math.exp

/**
 * Created by zhengchen on 14-7-21.
 */
object MathUtil {

  val rand = new Random(System.currentTimeMillis())

  def binomial(p: Double): Int = {
    if (p > rand.nextDouble())
      1
    else
      0
  }


  def multinomial(pa: Array[Double]): Int = {
    var sum = 0.0
    val cu = pa.map { a => sum += a; sum}
    val p = rand.nextDouble()
    cu.indexWhere(cumDist => cumDist >= p)
  }


  def softMax(a: Array[Double]): Array[Double] = {
    val total = a.map(exp(_)).sum
    a.map(exp(_)/total)
  }


}
